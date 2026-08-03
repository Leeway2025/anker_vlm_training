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


def save_ckpt(out_dir, train, opt_state, meta):
    """全量断点落盘: train 树 + optimizer 树(Adam μ/ν + MultiSteps 累积
    缓冲/计数)+ 进度元数据(step/best/patience/loss 历史)。

    崩溃安全: 先写 ckpt.tmp.npz 再 rename(POSIX 原子),滚动保留
    latest/prev 两份 —— "存到一半被杀"最坏丢一个保存间隔,不会留坏档。
    opt 树按 flatten 序号存(O:0000…): 结构由重建时的 optim.init 提供,
    序号只需与同配置下的 flatten 顺序一致(load_ckpt 有叶数硬校验)。
    返回耗时秒数。"""
    import json
    import os
    import time

    import jax
    t0 = time.time()

    def _raw(v):
        # bf16 等 ml_dtypes 在 npz 里不保真(读回变 void,jnp 无法 cast)
        # → 存原始位(同宽无符号整型视图),load 端按模板 dtype 视图还原
        v = np.asarray(v)
        if v.dtype.kind == "V":
            return v.view(f"u{v.dtype.itemsize}")
        return v

    tl = jax.tree_util.tree_flatten_with_path(train)[0]
    ol = jax.tree_util.tree_flatten(opt_state)[0]
    payload = {"__meta__": np.array(json.dumps(meta))}
    payload.update({"T:" + _path_str(p): _raw(v) for p, v in tl})
    payload.update({f"O:{i:04d}": _raw(v) for i, v in enumerate(ol)})
    tmp = os.path.join(out_dir, "ckpt.tmp.npz")
    latest = os.path.join(out_dir, "ckpt_latest.npz")
    prev = os.path.join(out_dir, "ckpt_prev.npz")
    np.savez(tmp, **payload)
    if os.path.exists(latest):
        os.replace(latest, prev)
    os.replace(tmp, latest)
    return time.time() - t0


def load_ckpt(out_dir, train0, opt_state0, jnp):
    """恢复断点。latest 损坏(截断/坏 zip)自动回退 prev;两份都没有
    返回 None(调用方从头开跑)。train0/opt_state0 只当结构+dtype 模板,
    值全部被断点覆盖。形状/叶数不匹配硬报错 —— 配置漂移必须显性失败。"""
    import json
    import os

    import jax

    def _typed(v, leaf, what):
        lt = np.dtype(leaf.dtype)
        if lt.kind == "V" and v.dtype.kind == "u" \
                and v.dtype.itemsize == lt.itemsize:
            v = v.view(lt)               # save 端 _raw 的逆: 原始位→ml_dtype
        if v.shape != np.shape(leaf):
            raise ValueError(f"{what} 形状 {v.shape} != 模板 "
                             f"{np.shape(leaf)}")
        return jnp.asarray(v, leaf.dtype)

    for name in ("ckpt_latest.npz", "ckpt_prev.npz"):
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            continue
        try:
            z = np.load(path, allow_pickle=False)
            meta = json.loads(str(z["__meta__"]))
            tl = jax.tree_util.tree_flatten_with_path(train0)[0]
            t_leaves = [_typed(z["T:" + _path_str(p)], leaf,
                               f"train 叶 {_path_str(p)}")
                        for p, leaf in tl]
            train = jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(train0), t_leaves)
            ol, odef = jax.tree_util.tree_flatten(opt_state0)
            n_saved = sum(1 for k in z.files if k.startswith("O:"))
            if n_saved != len(ol):
                raise ValueError(f"opt 叶数 {n_saved} != 模板 {len(ol)}"
                                 "(优化器配置变了?)")
            o_leaves = [_typed(z[f"O:{i:04d}"], ol[i], f"opt 叶 {i}")
                        for i in range(len(ol))]
            opt_state = jax.tree_util.tree_unflatten(odef, o_leaves)
            print(f"[resume] 载入 {name}: opt_step {meta.get('opt_step')} "
                  f"(train {len(t_leaves)} 叶 + opt {len(o_leaves)} 叶)")
            return train, opt_state, meta
        except Exception as e:  # noqa: BLE001 —— 坏档回退是本函数的职责
            print(f"[resume] {name} 不可用({e}),尝试上一份")
    return None
