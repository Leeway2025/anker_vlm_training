"""JAX SFT 训练循环 v2(多芯数据并行 + 视觉 LoRA + projector 全参)。

  /dev/shm/venv_jax/bin/python jax_impl/train_sft.py \
      --labels /dev/shm/fakedata/labels.jsonl --layout /dev/shm/hf_layout.json \
      --steps 10 --accum 2 --dp 4 --train-vision --train-projector

v1→v2:
  - shard_map 数据并行: 每设备沿用已验证的 bs1 路径,梯度 pmean;
  - --train-vision: VisionBlock nn.remat(纯数组签名,免闭包技巧)
    + 视觉 LoRA 解冻(v1 消融: 无 remat 的视觉反向吃 ~32G);
  - --train-projector: embedder/mm_input_projection(+norm)全参训练
    (对齐 torch 路线: projector 不在 adapter 内、单独训练);
  - 梯度裁剪 1.0(对齐 torch max_grad_norm)+ 每 K 步验证集 loss。
v1 的七个坑及修复见 FINDINGS.md;本文件保留全部关键注释。
"""
import argparse
import dataclasses
import functools
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--wds-dir", default=None,
                    help="显式指定分片目录(覆盖 labels 内 meta.wds_dir)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--dp", type=int, default=0, help="数据并行设备数,0=全部")
    ap.add_argument("--per-device-bs", type=int, default=1)
    ap.add_argument("--prefetch-workers", type=int, default=8,
                    help="0=关闭预取(同步取数,调试用)")
    ap.add_argument("--rank-scheme", choices=["uniform", "prod"],
                    default="uniform",
                    help="prod=生产方案: 差异化 rank 512/256 + rsLoRA α=2r")
    ap.add_argument("--lr", type=float, default=None,
                    help="缺省自动: prod=2e-5(v7 冷配方)/ uniform=1e-4")
    ap.add_argument("--proj-lr", type=float, default=5e-4)
    ap.add_argument("--vision-lr", type=float, default=2e-5,
                    help="视觉塔 LoRA 学习率(torch 生产: 2e-5)")
    ap.add_argument("--loraplus-ratio", type=float, default=None,
                    help="缺省自动: prod=1(v7 冷配方)/ uniform=16")
    ap.add_argument("--warmup", type=int, default=300,
                    help="warmup 步数(opt step;torch 生产: 300)")
    ap.add_argument("--lr-schedule", choices=["linear", "constant"],
                    default="linear",
                    help="linear=warmup+线性衰减到 0(torch Trainer 默认)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="torch 生产 wd=0;旧版隐含 optax 默认 1e-4")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle 与 val 切分种子")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--train-vision", action="store_true")
    ap.add_argument("--train-projector", action="store_true")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--val-n", type=int, default=8)
    ap.add_argument("--out", default="/dev/shm/out_jax_sft")
    ap.add_argument("--stage", choices=["a", "b"], default="b",
                    help="a=仅 projector 预热(LoRA 全冻结)")
    ap.add_argument("--sample-weights", help="hard-mining sw.json")
    ap.add_argument("--aux-file", help="资产 A attributes.jsonl → 7 属性头")
    ap.add_argument("--aux-conf-threshold", type=float, default=0.5,
                    help="低于此置信度的标注整条屏蔽(torch 同款)")
    ap.add_argument("--aux-coef", type=float, default=0.3)
    ap.add_argument("--ks-head", action="store_true", help="KS 父类头(6 类)")
    ap.add_argument("--ks-coef", type=float, default=0.2)
    ap.add_argument("--cot-file", help="资产 C reasoning.jsonl → 隐式 CoT")
    ap.add_argument("--cot-ratio", type=float, default=0.6)
    ap.add_argument("--cot-anneal", type=float, default=0.5,
                    help="最后该比例的步数切纯生产模式")
    ap.add_argument("--init-npz", help="从 train_params.npz 续训(或 import_hf 产物)")
    a = ap.parse_args()
    # v7 防过热: prod 的 rsLoRA scale(32/45)叠加热 lr/LoRA+ 会训死
    # 视觉→字母回路 —— prod 缺省自动用已验证冷配方,显式传参可覆盖
    # (覆盖越线时下方哨兵会警告)
    if a.lr is None:
        a.lr = 2e-5 if a.rank_scheme == "prod" else 1e-4
    if a.loraplus_ratio is None:
        a.loraplus_ratio = 1.0 if a.rank_scheme == "prod" else 16.0
    from jax_impl.logtee import tee_stdio
    tee_stdio(a.out)
    if a.stage == "a":
        a.train_projector = True     # stage a 语义: 只训 projector

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import flax.linen as nn
    from jax.sharding import Mesh, PartitionSpec as P
    from jax.experimental.shard_map import shard_map
    from gemma import gm
    from gemma import peft as gpeft
    from gemma.gm.nn.gemma4 import _modules as g4_modules
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from gemma.gm.nn.gemma4.vision import _transformer as gv_tr
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput

    from jax_impl.data import (SftDataset, make_vision_input,
                               install_batched_encode_vision)

    devs = jax.devices()
    DP = a.dp or len(devs)
    BS = a.per_device_bs
    print(f"[env] devices={len(devs)} dp={DP} per_device_bs={BS} "
          f"global_micro={DP*BS}")
    if BS > 1:
        install_batched_encode_vision()
    if a.rank_scheme == "prod":
        from jax_impl.prod_lora import install_prod_lora
        install_prod_lora()      # 必须在模型构造前(patch 参数创建路径)

    # ---- 逐层重算(gm 无内置 remat;v1 坑 3/4)----
    if not getattr(g4_modules, "_REMAT_PATCHED", False):
        _orig_call = g4_modules.Block.__call__
        _POL = jax.checkpoint_policies.nothing_saveable

        def _make_core(skip_flag):
            def core(self, x, pos, cache, mask, pli, kvs):
                return _orig_call(self, x, pos, cache, mask, pli, kvs,
                                  skip_sliding_mask=skip_flag)
            return nn.remat(core, policy=_POL, prevent_cse=False)

        _core = {False: _make_core(False), True: _make_core(True)}

        def patched(self, x, segment_pos, cache, attn_mask,
                    per_layer_input=None, kv_shared_cache=None,
                    skip_sliding_mask=False):
            return _core[bool(skip_sliding_mask)](
                self, x, segment_pos, cache, attn_mask,
                per_layer_input, kv_shared_cache)

        g4_modules.Block.__call__ = patched
        g4_modules._REMAT_PATCHED = True

    if a.train_vision and not getattr(gv_tr, "_REMAT_PATCHED", False):
        # 视觉塔反向无 remat 时吃 ~32G(v1 消融);VisionBlock 纯数组签名
        gv_tr.VisionBlock = nn.remat(
            gv_tr.VisionBlock,
            policy=jax.checkpoint_policies.nothing_saveable)
        gv_tr._REMAT_PATCHED = True
        print("[remat] VisionBlock 已包装")

    # HF 语义对齐(v1 坑 7,勿删): gm 训练路径 remove_mm_logits 会压缩
    # 视觉位 logits,尾部垃圾盖住 label 窗口 → 恒等旁路
    g4_tr._token_utils.remove_mm_logits = \
        lambda logits, tokens, num_tokens_per_image: logits

    # ---- 模型(视频语义: output_length=64,Gate C 配方)----
    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    model = gm.nn.LoRA(rank=a.rank,
                       model=gm.nn.Gemma4_E2B(text_only=False, config=cfg64))

    from jax_impl.data import load_jsonl_map
    tok = gm.text.Gemma4Tokenizer()
    sw = json.load(open(a.sample_weights)) if a.sample_weights else None
    # val_n 对齐 DP*BS: 旧版 val_n < DP*BS 时 eval 空转、val_loss 恒 0
    vn = 0
    if a.eval_every and a.val_n:
        g = DP * BS
        vn = ((a.val_n + g - 1) // g) * g
        if vn != a.val_n:
            print(f"[data] val_n {a.val_n} -> {vn}(对齐 DP*BS={g})")
    full = SftDataset(
        a.labels, a.layout, tok, wds_dir=a.wds_dir, sample_weights=sw,
        reasoning=load_jsonl_map(a.cot_file) if a.cot_file else None,
        cot_ratio=a.cot_ratio,
        attributes=load_jsonl_map(a.aux_file) if a.aux_file else None,
        seed=a.seed, val_n=vn, aux_conf_threshold=a.aux_conf_threshold)
    train_idx, val_idx = full.train_idx, full.val_idx
    print(f"[data] train={len(train_idx)} val={len(val_idx)}"
          f"(按 camera 切分, seed={a.seed}, 先切后复制) "
          f"max_len={full.max_len}")

    # ---- lora 结构初始化(v1 坑 1: eval_shape 免物化)----
    ex = full[0]
    p0, x0, counts = make_vision_input([ex["frames"]])
    dummy_pvi = PreprocessedVisionInput(
        patches=jnp.asarray(p0), positions_xy=jnp.asarray(x0),
        soft_token_counts=counts)
    struct = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0),
        tokens=jnp.asarray(ex["tokens"][None]), images=dummy_pvi))
    _, lora_struct = gpeft.split_params(struct["params"])
    rng = np.random.RandomState(0)

    def _path_str(path):
        return "/".join(getattr(k, "key", str(k)) for k in path)

    # 可训集合 ≡ 可交付集合(torch prod targets / export_hf 映射):
    # gm.nn.LoRA 会给 per_layer_input_gate 等 PLE 模块也注入 LoRA,但
    # torch 生产不适配、HF 无导出映射 —— 训了也交付不了,还造成
    # JAX 评测与端侧成品的偏差,一律冻结(2026-07-20)
    EXPORTABLE = ("q_einsum", "kv_einsum", "attn_vec_einsum",
                  "gating_einsum", "/mlp/linear")

    def _is_trainable(pstr):
        if a.stage == "a":
            return False                     # stage a: LoRA 全冻结
        llm = pstr.startswith("layer_") or "/layer_" in pstr
        vis = (a.train_vision and "vision_encoder" in pstr
               and "stacked_layers" in pstr)  # entry 投影无 HF 注入点,不训
        if not (llm or vis):
            return False
        return any(k in pstr for k in EXPORTABLE)

    def init_leaf(path, leaf):
        pstr = _path_str(path)
        if pstr.endswith("/a") and _is_trainable(pstr):
            # peft gaussian 语义: std=1/r(r=叶末维)。旧值 0.02 偏大
            # 5-10×,叠加 rsLoRA scale 32/45 与 LoRA+ B×16 后有效步长
            # 超 torch 一个量级 → 长程视频→字母回路被打饱和,训练收敛
            # 到常数字母(2026-07-21 过拟合消融实锤: prod 常数/uniform
            # 100%;修复后 prod 亦 100%,见 FINDINGS v7)
            std = 1.0 / leaf.shape[-1]
            return jnp.asarray(rng.normal(0, std, leaf.shape), jnp.float32)
        return jnp.zeros(leaf.shape, jnp.float32)   # B 零 + 冻结项零
    lora0 = jax.tree_util.tree_map_with_path(init_leaf, lora_struct)

    base = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)

    # ---- projector 全参(从 base 抽出作为可训练树)----
    PROJ_KEYS = ("mm_input_projection",)   # checkpoint 实测仅此一项(与 torch projector tensors=1 一致)
    if a.train_projector:
        proj0 = {k: jax.tree.map(lambda x: jnp.asarray(x, jnp.float32),
                                 base["embedder"][k]) for k in PROJ_KEYS}
        print(f"[proj] 全参训练: {list(proj0)} "
              f"({sum(x.size for x in jax.tree.leaves(proj0))/1e6:.1f}M)")
    else:
        proj0 = {}
    base = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), base)

    # ---- 辅助头(7 属性)与 KS 父类头: 独立 fp32 小参数树 ----
    from data.taxonomy import AUX_VOCABS, AUX_HEAD_ORDER, KS_CLASSES
    D_MODEL = 1536
    aux0 = {}
    if a.aux_file:
        for h in AUX_HEAD_ORDER:
            n_cls = len(AUX_VOCABS[h])
            aux0[h] = {"w": jnp.asarray(
                rng.normal(0, 0.02, (D_MODEL, n_cls)), jnp.float32),
                "b": jnp.zeros((n_cls,), jnp.float32)}
    if a.ks_head:
        aux0["ks"] = {"w": jnp.asarray(
            rng.normal(0, 0.02, (D_MODEL, len(KS_CLASSES))), jnp.float32),
            "b": jnp.zeros((len(KS_CLASSES),), jnp.float32)}

    train0 = {"lora": lora0, "proj": proj0, "aux": aux0}
    if a.init_npz:                        # 续训: 覆盖同名叶(形状须一致)
        from jax_impl.npz_io import restore_train_tree
        z = np.load(a.init_npz)
        # ③号致命坑防护: npz 里的全零 LoRA a(如 stage a 产物,LoRA 全程
        # 冻结为零)不得覆盖新初始化 —— A=0∧B=0 梯度恒零,恢复即杀死适配器
        train0, st = restore_train_tree(
            train0, z, jnp, is_zero_skippable=_is_trainable)
        msg = f"[init-npz] 恢复 {st['hit']} 叶 from {a.init_npz}"
        if st["shape_skip"]:
            msg += (f";形状不符跳过 {st['shape_skip']} 叶"
                    f"(跨 rank 方案衔接时属预期,如 S1 uniform→S2 prod)")
        if st["zero_a_skip"]:
            msg += (f";⚠️ 全零 LoRA a 跳过 {st['zero_a_skip']} 叶"
                    f"(npz 来自未训 LoRA 的阶段,保留本次新初始化)")
        print(msg)
    # lr 日程: warmup + 线性衰减到 0(对齐 torch Trainer 默认;MultiSteps
    # 只在累积满时调用内层 update → 日程按 opt step 计数,与 torch 同拍)
    def mk_sched(peak):
        if a.lr_schedule == "constant":
            return peak
        return optax.join_schedules(
            [optax.linear_schedule(0.0, peak, max(a.warmup, 1)),
             optax.linear_schedule(peak, 0.0, max(a.steps - a.warmup, 1))],
            [max(a.warmup, 1)])

    # 分组对齐 torch 生产: vision LoRA 2e-5(旧版错用 1e-4,5×超速);
    # LoRA+ B 矩阵 lr×16;aux 头随 LLM-A 组;wd 显式 0(optax 默认 1e-4
    # 是 torch 没有的隐藏收缩力)
    def _group(p, _):
        k = _path_str(p)
        if k.startswith("proj"):
            return "proj"
        vis = "vision_encoder" in k
        b = k.endswith("/b")
        if vis:
            return "vis_b" if b else "vis_a"
        return "llm_b" if b else "llm_a"
    GROUP_LR = {"proj": a.proj_lr,
                "llm_a": a.lr, "llm_b": a.lr * a.loraplus_ratio,
                "vis_a": a.vision_lr,
                "vis_b": a.vision_lr * a.loraplus_ratio}
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),         # 对齐 torch max_grad_norm
        optax.multi_transform(
            {g: optax.adamw(mk_sched(lr), weight_decay=a.weight_decay)
             for g, lr in GROUP_LR.items()},
            param_labels=lambda tree: jax.tree_util.tree_map_with_path(
                _group, tree)))
    if a.rank_scheme == "prod" and a.lr * a.loraplus_ratio > 1e-4:
        print("⚠️ [optim] prod 方案 + 高 lr/LoRA+ 组合: rsLoRA scale(32/45)"
              "叠加后有效步长过热,实测会把视觉→字母回路训死"
              "(FINDINGS v7 过拟合消融)。已验证配方: --lr 2e-5 "
              "--loraplus-ratio 1")
    print(f"[optim] {a.lr_schedule} warmup={a.warmup} wd={a.weight_decay} "
          f"lr: llm_a={a.lr:g} llm_b={a.lr*a.loraplus_ratio:g} "
          f"vis_a={a.vision_lr:g} vis_b={a.vision_lr*a.loraplus_ratio:g} "
          f"proj={a.proj_lr:g}")
    optim = optax.MultiSteps(tx, every_k_schedule=a.accum)
    opt_state = optim.init(train0)

    ls = a.label_smoothing
    T = len(full.template)
    mesh = Mesh(np.asarray(devs[:DP]), ("dp",))

    def loss_fn(train, base_p, tokens, labels, weights, patches, pos_xy,
                aux_labels, ks_label):
        # v1 坑 5: fp32 参与会把激活链提升 fp32 → 前向统一 bf16;
        # 冻结 lora 叶 stop_gradient(v1 坑 6: 切断未训练子树反向)
        lora_h = jax.tree_util.tree_map_with_path(
            lambda p, x: x.astype(jnp.bfloat16)
            if _is_trainable(_path_str(p))
            else jax.lax.stop_gradient(x.astype(jnp.bfloat16)),
            train["lora"])
        base_h = base_p
        if a.train_projector:
            emb = dict(base_p["embedder"])
            for k in PROJ_KEYS:
                emb[k] = jax.tree.map(
                    lambda x: x.astype(jnp.bfloat16), train["proj"][k])
            base_h = dict(base_p)
            base_h["embedder"] = emb
        params = gpeft.merge_params(base_h, lora_h)
        pvi = PreprocessedVisionInput(
            patches=patches, positions_xy=pos_xy, soft_token_counts=counts)
        need_hidden = bool(train["aux"])
        out = model.apply({"params": params}, tokens=tokens, images=pvi,
                          return_hidden_states=need_hidden or None)
        logits = out.logits if hasattr(out, "logits") else out
        lg = logits[:, T - 1:-1].astype(jnp.float32)   # 尾窗(v1 坑 2)
        lb = labels[:, T:]
        wt = weights[:, T:]
        valid = (lb != -100).astype(jnp.float32)
        lse = jax.nn.logsumexp(lg, axis=-1)
        tgt = jnp.take_along_axis(
            lg, jnp.clip(lb, 0)[..., None], axis=-1)[..., 0]
        ce = (1 - ls) * (lse - tgt) + ls * (lse - lg.mean(-1))
        ce = jnp.where(valid > 0, ce, 0.0)      # PAD 行 NaN×0 防护
        loss = (ce * wt).sum() / jnp.clip((wt * valid).sum(), 1.0)

        if need_hidden:
            hs = out.hidden_states
            hs = hs[-1] if isinstance(hs, (tuple, list)) else hs
            # 视觉位池化(哨兵 -2 位置): 属性/KS 都是视觉判断
            vmask = (tokens == -2).astype(jnp.float32)[..., None]
            pooled = ((hs.astype(jnp.float32) * vmask).sum(1)
                      / jnp.clip(vmask.sum(1), 1.0))            # [B, D]
            def head_ce(w, b, y):
                lgt = pooled @ w + b
                lp = jax.nn.log_softmax(lgt)
                ok = (y >= 0).astype(jnp.float32)
                pick = jnp.take_along_axis(
                    lp, jnp.clip(y, 0)[:, None], axis=-1)[:, 0]
                return -(pick * ok).sum() / jnp.clip(ok.sum(), 1.0)
            aux_l = 0.0
            n_heads = 0
            from data.taxonomy import AUX_HEAD_ORDER as _AHO
            for j, h in enumerate(_AHO):
                if h in train["aux"]:
                    aux_l += head_ce(train["aux"][h]["w"],
                                     train["aux"][h]["b"], aux_labels[:, j])
                    n_heads += 1
            if n_heads:
                loss = loss + a.aux_coef * aux_l / n_heads
            if "ks" in train["aux"]:
                loss = loss + a.ks_coef * head_ce(
                    train["aux"]["ks"]["w"], train["aux"]["ks"]["b"], ks_label)
        return loss

    def grad_local(train, base_p, tokens, labels, weights, patches, pos_xy,
                   aux_labels, ks_label):
        loss, grads = jax.value_and_grad(loss_fn)(
            train, base_p, tokens, labels, weights, patches, pos_xy,
            aux_labels, ks_label)
        return (jax.lax.pmean(loss, "dp"),
                jax.tree.map(lambda g: jax.lax.pmean(g, "dp"), grads))

    grad_sharded = shard_map(
        grad_local, mesh=mesh,
        in_specs=(P(), P(), P("dp"), P("dp"), P("dp"), P("dp"), P("dp"),
                  P("dp"), P("dp")),
        out_specs=(P(), P()), check_rep=False)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def train_step(train, opt_state, base_p, tokens, labels, weights,
                   patches, pos_xy, aux_labels, ks_label):
        loss, grads = grad_sharded(train, base_p, tokens, labels, weights,
                                   patches, pos_xy, aux_labels, ks_label)
        updates, opt_state = optim.update(grads, opt_state, train)
        train = optax.apply_updates(train, updates)
        return train, opt_state, loss

    eval_local = shard_map(
        lambda tr, bp, t, l, w, p, x, al, kl: jax.lax.pmean(
            loss_fn(tr, bp, t, l, w, p, x, al, kl), "dp"),
        mesh=mesh,
        in_specs=(P(), P(), P("dp"), P("dp"), P("dp"), P("dp"), P("dp"),
                  P("dp"), P("dp")),
        out_specs=P(), check_rep=False)
    eval_loss_j = jax.jit(eval_local)

    def collect(idxs):
        exs = [full[i] for i in idxs]
        pt_all, px_all, _ = make_vision_input([e["frames"] for e in exs])
        pt, px = list(pt_all), list(px_all)
        return (jnp.asarray(np.stack([e["tokens"] for e in exs])),
                jnp.asarray(np.stack([e["labels"] for e in exs])),
                jnp.asarray(np.stack([e["weights"] for e in exs])),
                jnp.asarray(np.stack(pt)), jnp.asarray(np.stack(px)),
                jnp.asarray(np.stack([e["aux_labels"] for e in exs])),
                jnp.asarray(np.stack([e["ks_label"] for e in exs])))

    train = train0
    os.makedirs(a.out, exist_ok=True)
    hist, best = [], (1e9, -1)
    cursor = 0
    t0 = time.time()
    total_micro = a.steps * a.accum
    # 每 epoch 用固定 seed 重洗(旧版严格顺序循环: 同类样本成段 →
    # batch 内梯度强相关;hard-mining 副本相邻 → 等效单步 lr×n)。
    # 全部置换预生成 → 线程安全、prefetch/同步路径逐条一致、可复现
    ep_len = len(train_idx)
    _rs = np.random.RandomState(a.seed)
    _perms = [_rs.permutation(ep_len)
              for _ in range(total_micro * DP * BS // ep_len + 2)]
    train_np = np.asarray(train_idx)

    def draw(k):
        e, i = divmod(k, ep_len)
        return int(train_np[_perms[e][i]])
    switch_at = int(total_micro * (1 - a.cot_anneal)) if a.cot_file else -1
    pf = None
    if a.prefetch_workers > 0:
        from jax_impl.prefetch import BatchPrefetcher
        pf = BatchPrefetcher(full, draw,
                             DP * BS, workers=a.prefetch_workers)
        print(f"[prefetch] workers={a.prefetch_workers} depth=2 "
              f"(每 epoch 重洗, seed={a.seed})")
    for micro in range(total_micro):
        if switch_at >= 0 and micro == switch_at and not full.anneal:
            full.set_anneal(True)
            if pf:                      # 清掉队列里旧模式 batch,边界零滞后
                pf.flush(restart_at=micro * DP * BS)
            print(f"[anneal] micro {micro}: 切换纯生产模式", flush=True)
        if pf:
            t_, l_, w_, p_, x_, al_, kl_, _ = pf.next()
            batch = tuple(jnp.asarray(v)
                          for v in (t_, l_, w_, p_, x_, al_, kl_))
        else:
            idxs = [draw(cursor + j) for j in range(DP * BS)]
            cursor += DP * BS
            batch = collect(idxs)
        train, opt_state, loss = train_step(train, opt_state, base, *batch)
        if micro == 0:
            print(f"[compile+step0] {time.time()-t0:.0f}s", flush=True)
            t0 = time.time()
        if (micro + 1) % a.accum == 0:
            opt_step = (micro + 1) // a.accum
            l = float(loss)              # 强制同步 → 边际耗时是真实的
            hist.append(l)
            now = time.time()
            dt = (now - getattr(main, "_tprev", t0)) / a.accum
            main._tprev = now
            print(f"[sft] opt_step {opt_step}/{a.steps} loss={l:.4f} "
                  f"marginal_micro_s={dt:.3f} "
                  f"samples/s={DP*BS/max(dt,1e-9):.1f}", flush=True)
            if a.eval_every and opt_step % a.eval_every == 0:
                vl = []
                for k in range(0, len(val_idx) - DP * BS + 1, DP * BS):
                    vb = collect(val_idx[k:k + DP * BS])
                    vl.append(float(eval_loss_j(train, base, *vb)))
                v = sum(vl) / max(len(vl), 1)
                tag = ""
                if vl and v < best[0]:
                    best = (v, opt_step); tag = " *best"
                    # best 即时落盘 —— 旧版只记 meta,交付的永远是最后
                    # 一步(过拟合了也照存)
                    bf = jax.tree_util.tree_flatten_with_path(train)[0]
                    np.savez(os.path.join(a.out, "train_params_best.npz"),
                             **{_path_str(p): np.asarray(x) for p, x in bf})
                print(f"[eval] opt_step {opt_step} val_loss={v:.4f}{tag}",
                      flush=True)

    try:
        ms = jax.local_devices()[0].memory_stats()
        print(f"[hbm] dev0 peak={ms.get('peak_bytes_in_use', 0)/2**30:.2f}G "
              f"limit={ms.get('bytes_limit', 0)/2**30:.2f}G")
    except Exception:  # noqa: BLE001
        pass
    if pf:
        pf.close()
    flat = jax.tree_util.tree_flatten_with_path(train)[0]
    np.savez(os.path.join(a.out, "train_params.npz"),
             **{_path_str(p): np.asarray(v) for p, v in flat})
    from jax_impl.logtee import code_version
    json.dump({"loss_history": hist, "rank": a.rank, "dp": DP,
               "best_val": list(best), "seed": a.seed,
               "lr_schedule": a.lr_schedule, "warmup": a.warmup,
               "code_commit": code_version()},
              open(os.path.join(a.out, "train_meta.json"), "w"))
    has_best = os.path.exists(os.path.join(a.out, "train_params_best.npz"))
    print(f"[save] {a.out} (loss {hist[0]:.3f} -> {hist[-1]:.3f}, "
          f"best_val={best[0]:.4f}@{best[1]})"
          + ("\n[save] 评测/交付请用 train_params_best.npz"
             f"(val 最优 @step {best[1]});train_params.npz 是最后一步"
             if has_best else ""))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
