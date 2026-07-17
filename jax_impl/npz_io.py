"""训练产物 npz 的加载/校验(infer/kto/train_sft 共用)。

三个致命坑的集中修复点(2026-07 review):
  ① infer/kto 猜 rank + shape 不匹配静默填零 → prod 产物评测≈base;
     修复: rank 方案从 npz 自动判定 + 严格加载(任何未命中即硬报错)。
  ② proj/ 子树(训练过的 projector)从不合并进 base → 训练/推理模型不一致;
     修复: merge_proj_into_base,缺失时按调用方要求报错或告警。
  ③ stage a 产物携带全零 LoRA 树,同方案续训会把新初始化覆盖为零
     → LoRA 梯度恒零(A=0 且 B=0 是死鞍点);
     修复: restore 跳过"npz 全零且本地已有非零初始化"的 a 叶并告警。
"""
import numpy as np


def _path_str(path):
    return "/".join(getattr(k, "key", str(k)) for k in path)


def detect_rank_scheme(z):
    """从 npz 推断 LoRA rank 方案。

    返回 ("prod", ranks) 或 ("uniform", ranks)。依据: LLM 层 a 叶的
    rank 集合 —— 差异化({256,512} 等)即 prod,单一即 uniform。
    npz 无 LLM LoRA 键时报错(产物损坏或根本不是训练产物)。
    """
    ranks = sorted({int(z[k].shape[-1]) for k in z.files
                    if k.startswith("lora/") and k.endswith("/a")
                    and "layer_" in k})
    if not ranks:
        raise ValueError("npz 中无 LLM LoRA 键(lora/…layer_…/a)—— "
                         "不是训练产物,或保存被截断")
    return ("prod" if len(ranks) > 1 else "uniform"), ranks


def load_lora_strict(z, lora_struct, jnp, dtype):
    """按结构树从 npz 严格加载 LoRA: 任何叶未命中/形状不符 → 硬报错。

    静默填零是①号致命坑的根源: prod 产物(global 层 r=512)被 uniform
    模型结构加载时 7 个 global 层直接归零、无任何告警。这里改为:
    结构叶必须逐一在 npz 中命中,否则列出示例并报错(大概率是
    --rank-scheme 与训练时不一致)。
    """
    import jax
    missing = []

    def fill(path, leaf):
        k = "lora/" + _path_str(path)
        if k in z.files and z[k].shape == leaf.shape:
            return jnp.asarray(z[k], dtype)
        missing.append((k, tuple(leaf.shape),
                        tuple(z[k].shape) if k in z.files else None))
        return jnp.zeros(leaf.shape, dtype)

    lora = jax.tree_util.tree_map_with_path(fill, lora_struct)
    if missing:
        ex = "; ".join(f"{k} 期望{s} npz={zs}" for k, s, zs in missing[:4])
        raise ValueError(
            f"LoRA 加载未命中 {len(missing)} 叶(示例: {ex})—— "
            f"rank 方案与训练时不一致?训练产物用 --rank-scheme 同款重试")
    n_nonzero = sum(1 for v in jax.tree.leaves(lora)
                    if np.abs(np.asarray(v)).max() > 0)
    if n_nonzero == 0:
        raise ValueError("npz 中 LoRA 全零 —— 该产物的训练阶段未训 LoRA"
                         "(如 stage a),不能作为推理/KTO 的 LoRA 来源")
    return lora


def merge_proj_into_base(base_params, z, jnp, dtype, required):
    """把 npz 的 proj/ 子树(训练过的 projector)写回 base_params[embedder]。

    ②号致命坑: stage a/b 训的 projector 在推理/KTO 被静默丢弃。
    required=True 时 npz 无 proj/ 键即报错(--train-projector 产物必有)。
    返回 (params, 写回叶数)。
    """
    proj_keys = [k for k in z.files if k.startswith("proj/")]
    if not proj_keys:
        if required:
            raise ValueError("npz 中无 proj/ 子树 —— 训练时未开 "
                             "--train-projector?若确认无需 projector,"
                             "传 --no-proj 跳过")
        return base_params, 0
    emb = dict(base_params["embedder"])
    for k in proj_keys:
        segs = k.split("/")[1:]              # 去掉前缀 proj
        node, parents = emb, []
        for s in segs[:-1]:
            parents.append((node, s))
            node = dict(node[s])
        leaf = segs[-1]
        if leaf not in node or node[leaf].shape != z[k].shape:
            raise ValueError(f"proj 键 {k} 在 base 中无对应位置或形状不符 "
                             f"(base={node.get(leaf) is not None and node[leaf].shape})")
        node[leaf] = jnp.asarray(z[k], dtype)
        for parent, s in reversed(parents):
            parent[s] = node
            node = parent
    base = dict(base_params)
    base["embedder"] = emb
    return base, len(proj_keys)


def restore_train_tree(train0, z, jnp, is_zero_skippable):
    """train_sft --init-npz 续训恢复(修③号坑 + n_hit 如实报账)。

    规则: key+shape 命中才恢复;lora a 叶若 npz 值全零、而本地初始化
    非零且该叶可训练(is_zero_skippable(pstr)=True)→ 跳过恢复并告警
    (A=0 且 B=0 的 LoRA 梯度恒零,恢复它等于杀死整个适配器;
    全零 a 不携带任何训练信息,跳过在数学上严格无损)。
    返回 (树, 统计 dict)。
    """
    import jax
    flat = dict(z)
    stats = {"hit": 0, "shape_skip": 0, "zero_a_skip": 0}

    def restore(path, leaf):
        k = _path_str(path)
        if k not in flat:
            return leaf
        if flat[k].shape != leaf.shape:
            stats["shape_skip"] += 1
            return leaf
        if (k.startswith("lora/") and k.endswith("/a")
                and is_zero_skippable(k)
                and np.abs(flat[k]).max() == 0
                and np.abs(np.asarray(leaf)).max() > 0):
            stats["zero_a_skip"] += 1
            return leaf
        stats["hit"] += 1
        return jnp.asarray(flat[k], leaf.dtype)

    tree = jax.tree_util.tree_map_with_path(restore, train0)
    return tree, stats
