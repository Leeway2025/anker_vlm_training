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
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--dp", type=int, default=0, help="数据并行设备数,0=全部")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--proj-lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--train-vision", action="store_true")
    ap.add_argument("--train-projector", action="store_true")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--val-n", type=int, default=8)
    ap.add_argument("--out", default="/dev/shm/out_jax_sft")
    a = ap.parse_args()

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

    from jax_impl.data import SftDataset, make_vision_input

    devs = jax.devices()
    DP = a.dp or len(devs)
    print(f"[env] devices={len(devs)} dp={DP}")

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

    tok = gm.text.Gemma4Tokenizer()
    full = SftDataset(a.labels, a.layout, tok)
    val_idx = list(range(len(full)))[-a.val_n:]
    train_idx = [i for i in range(len(full)) if i not in set(val_idx)]
    print(f"[data] train={len(train_idx)} val={len(val_idx)} "
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

    def _is_trainable(pstr):
        if pstr.startswith("layer_") or "/layer_" in pstr:
            return True                      # LLM attn+mlp(v1 同范围)
        if a.train_vision and "vision_encoder" in pstr:
            return True
        return False

    def init_leaf(path, leaf):
        pstr = _path_str(path)
        if pstr.endswith("/a") and _is_trainable(pstr):
            return jnp.asarray(rng.normal(0, 0.02, leaf.shape), jnp.float32)
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

    train0 = {"lora": lora0, "proj": proj0}
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),         # 对齐 torch max_grad_norm
        optax.multi_transform(
            {"lora": optax.adamw(a.lr), "proj": optax.adamw(a.proj_lr)},
            param_labels=lambda tree: jax.tree_util.tree_map_with_path(
                lambda p, _: "proj" if _path_str(p).startswith("proj")
                else "lora", tree)))
    optim = optax.MultiSteps(tx, every_k_schedule=a.accum)
    opt_state = optim.init(train0)

    ls = a.label_smoothing
    T = len(full.template)
    mesh = Mesh(np.asarray(devs[:DP]), ("dp",))

    def loss_fn(train, base_p, tokens, labels, weights, patches, pos_xy):
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
        out = model.apply({"params": params}, tokens=tokens, images=pvi)
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
        return (ce * wt).sum() / jnp.clip((wt * valid).sum(), 1.0)

    def grad_local(train, base_p, tokens, labels, weights, patches, pos_xy):
        loss, grads = jax.value_and_grad(loss_fn)(
            train, base_p, tokens, labels, weights, patches, pos_xy)
        return (jax.lax.pmean(loss, "dp"),
                jax.tree.map(lambda g: jax.lax.pmean(g, "dp"), grads))

    grad_sharded = shard_map(
        grad_local, mesh=mesh,
        in_specs=(P(), P(), P("dp"), P("dp"), P("dp"), P("dp"), P("dp")),
        out_specs=(P(), P()), check_rep=False)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def train_step(train, opt_state, base_p, tokens, labels, weights,
                   patches, pos_xy):
        loss, grads = grad_sharded(train, base_p, tokens, labels, weights,
                                   patches, pos_xy)
        updates, opt_state = optim.update(grads, opt_state, train)
        train = optax.apply_updates(train, updates)
        return train, opt_state, loss

    eval_local = shard_map(
        lambda tr, bp, t, l, w, p, x: jax.lax.pmean(
            loss_fn(tr, bp, t, l, w, p, x), "dp"),
        mesh=mesh,
        in_specs=(P(), P(), P("dp"), P("dp"), P("dp"), P("dp"), P("dp")),
        out_specs=P(), check_rep=False)
    eval_loss_j = jax.jit(eval_local)

    def collect(idxs):
        exs = [full[i] for i in idxs]
        pt, px = [], []
        for e in exs:
            pp, xx, _ = make_vision_input([e["frames"]])
            pt.append(pp[0]); px.append(xx[0])
        return (jnp.asarray(np.stack([e["tokens"] for e in exs])),
                jnp.asarray(np.stack([e["labels"] for e in exs])),
                jnp.asarray(np.stack([e["weights"] for e in exs])),
                jnp.asarray(np.stack(pt)), jnp.asarray(np.stack(px)))

    train = train0
    os.makedirs(a.out, exist_ok=True)
    hist, best = [], (1e9, -1)
    cursor = 0
    t0 = time.time()
    for micro in range(a.steps * a.accum):
        idxs = [train_idx[(cursor + j) % len(train_idx)] for j in range(DP)]
        cursor += DP
        batch = collect(idxs)
        train, opt_state, loss = train_step(train, opt_state, base, *batch)
        if micro == 0:
            print(f"[compile+step0] {time.time()-t0:.0f}s", flush=True)
            t0 = time.time()
        if (micro + 1) % a.accum == 0:
            opt_step = (micro + 1) // a.accum
            l = float(loss)
            hist.append(l)
            n_micro = max(opt_step * a.accum - 1, 1)
            dt = (time.time() - t0) / n_micro
            print(f"[sft] opt_step {opt_step}/{a.steps} loss={l:.4f} "
                  f"micro_s/it={dt:.2f} samples/s={DP/dt:.2f}", flush=True)
            if a.eval_every and opt_step % a.eval_every == 0:
                vl = []
                for k in range(0, len(val_idx) - DP + 1, DP):
                    vb = collect(val_idx[k:k + DP])
                    vl.append(float(eval_loss_j(train, base, *vb)))
                v = sum(vl) / max(len(vl), 1)
                tag = ""
                if v < best[0]:
                    best = (v, opt_step); tag = " *best"
                print(f"[eval] opt_step {opt_step} val_loss={v:.4f}{tag}",
                      flush=True)

    try:
        ms = jax.local_devices()[0].memory_stats()
        print(f"[hbm] dev0 peak={ms.get('peak_bytes_in_use', 0)/2**30:.2f}G "
              f"limit={ms.get('bytes_limit', 0)/2**30:.2f}G")
    except Exception:  # noqa: BLE001
        pass
    flat = jax.tree_util.tree_flatten_with_path(train)[0]
    np.savez(os.path.join(a.out, "train_params.npz"),
             **{_path_str(p): np.asarray(v) for p, v in flat})
    json.dump({"loss_history": hist, "rank": a.rank, "dp": DP,
               "best_val": list(best)},
              open(os.path.join(a.out, "train_meta.json"), "w"))
    print(f"[save] {a.out} (loss {hist[0]:.3f} -> {hist[-1]:.3f}, "
          f"best_val={best[0]:.4f}@{best[1]})")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
