"""KTO 偏好优化(JAX 版;损失数学对齐 torch training/kto.py = TRL KTOTrainer)。

  python jax_impl/kto.py --kto-data kto_data.jsonl --labels labels.jsonl \
      --layout hf_layout.json --init-npz sft_lora.npz --steps 10

与 torch 版的结构差异(简化而语义等价):
  - reference = base(冻结底座)。torch 用"初始 adapter 副本"作 ref,
    而初始 adapter B=0 ⇒ ref ≡ base,JAX 直接省掉 set_adapter 切换;
  - 多芯数据并行(--dp,0=全部): shard_map 每设备一对样本,梯度
    pmean;主循环固定步数,各设备程序一致 → 无 all_reduce 不齐风险
    (torch 侧分片不齐死锁的教训);
  - 错配 KL: batch 维滚动(与 torch roll_completions 同语义)——
    kl 对 i = (video_i, completion_{i-1 mod DP})。
"""
import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EOT = 106


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kto-data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--wds-dir", default=None,
                    help="显式指定分片目录(覆盖 labels 内 meta.wds_dir)")
    ap.add_argument("--init-npz", help="SFT 产物 lora(policy 起点)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--w-desirable", type=float, default=1.0)
    ap.add_argument("--w-undesirable", type=float, default=1.5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--completion-budget", type=int, default=64)
    ap.add_argument("--dp", type=int, default=0, help="数据并行设备数,0=全部")
    ap.add_argument("--out", default="/dev/shm/out_jax_kto")
    a = ap.parse_args()
    from jax_impl.logtee import tee_stdio
    tee_stdio(a.out)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import flax.linen as nn
    from gemma import gm
    from gemma import peft as gpeft
    from gemma.gm.nn.gemma4 import _modules as g4_modules
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
    from jax_impl.data import SftDataset, make_vision_input, load_frames

    # 与 train_sft 相同的三个环境补丁(remat / remove_mm_logits)
    if not getattr(g4_modules, "_REMAT_PATCHED", False):
        _orig = g4_modules.Block.__call__
        _POL = jax.checkpoint_policies.nothing_saveable

        def _mk(flag):
            def core(self, x, pos, cache, mask, pli, kvs):
                return _orig(self, x, pos, cache, mask, pli, kvs,
                             skip_sliding_mask=flag)
            return nn.remat(core, policy=_POL, prevent_cse=False)
        _c = {False: _mk(False), True: _mk(True)}
        def _patched(self, x, segment_pos, cache, attn_mask,
                     per_layer_input=None, kv_shared_cache=None,
                     skip_sliding_mask=False):
            return _c[bool(skip_sliding_mask)](
                self, x, segment_pos, cache, attn_mask,
                per_layer_input, kv_shared_cache)
        g4_modules.Block.__call__ = _patched
        g4_modules._REMAT_PATCHED = True
    g4_tr._token_utils.remove_mm_logits = \
        lambda logits, tokens, num_tokens_per_image: logits

    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    model = gm.nn.LoRA(rank=a.rank,
                       model=gm.nn.Gemma4_E2B(text_only=False, config=cfg64))
    tok = gm.text.Gemma4Tokenizer()
    ds = SftDataset(a.labels, a.layout, tok, wds_dir=a.wds_dir,
                    max_label_len=a.completion_budget)
    T = len(ds.template)
    K = a.completion_budget
    L = T + K
    recs_by_vid = {r["video_id"]: i for i, r in enumerate(ds.recs)}

    pairs = [json.loads(l) for l in open(a.kto_data, encoding="utf-8")]
    pairs = [p for p in pairs if p["video_id"] in recs_by_vid]
    print(f"[kto] pairs={len(pairs)} "
          f"des={sum(1 for p in pairs if p['label'])} "
          f"und={sum(1 for p in pairs if not p['label'])}")

    # ---- 参数: base 冻结 + lora(policy 可训练;ref = base 本身)----
    ex0 = ds[0]
    p0, x0, counts = make_vision_input([ex0["frames"]])
    struct = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0), tokens=jnp.zeros((1, L), jnp.int32),
        images=PreprocessedVisionInput(
            patches=jnp.asarray(p0), positions_xy=jnp.asarray(x0),
            soft_token_counts=counts)))
    lora_struct = gpeft.split_params(struct["params"])[1]
    rng = np.random.RandomState(0)
    z = np.load(a.init_npz) if a.init_npz else None

    def init_leaf(path, leaf):
        k = "/".join(getattr(x, "key", str(x)) for x in path)
        if z is not None:
            zk = "lora/" + k
            if zk in z.files and z[zk].shape == leaf.shape:
                return jnp.asarray(z[zk], jnp.float32)
        llm = k.startswith("layer_") or "/layer_" in k
        if k.endswith("/a") and llm:
            return jnp.asarray(rng.normal(0, 0.02, leaf.shape), jnp.float32)
        return jnp.zeros(leaf.shape, jnp.float32)
    lora0 = jax.tree_util.tree_map_with_path(init_leaf, lora_struct)

    base = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    base = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), base)

    optim = optax.MultiSteps(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(a.lr)),
        every_k_schedule=a.accum)
    opt_state = optim.init(lora0)

    def build(pair):
        i = recs_by_vid[pair["video_id"]]
        ex = ds[i]                      # 只用其 frames/模板
        comp = tok.encode(pair["completion"].strip()) + [EOT]
        comp = comp[:K]
        toks = np.zeros(L, np.int32); toks[:T] = ds.template
        toks[T:T + len(comp)] = comp
        labs = np.full(L, -100, np.int32)
        labs[T:T + len(comp)] = comp
        pt, px, _ = make_vision_input([ex["frames"]])
        return (toks, labs, pt[0], px[0], 1.0 if pair["label"] else 0.0)

    def sum_logprob_fn(params, tokens, labels, pvi):
        out = model.apply({"params": params}, tokens=tokens, images=pvi)
        lg = (out.logits if hasattr(out, "logits") else out)
        lg = lg[:, T - 1:-1].astype(jnp.float32)
        lb = labels[:, T:]
        m = (lb != -100).astype(jnp.float32)
        lse = jax.nn.logsumexp(lg, axis=-1)
        tgt = jnp.take_along_axis(
            lg, jnp.clip(lb, 0)[..., None], axis=-1)[..., 0]
        lp = jnp.where(m > 0, tgt - lse, 0.0)
        return lp.sum(axis=1)                      # [B]

    def _is_llm(path):
        k = "/".join(getattr(x, "key", str(x)) for x in path)
        return k.startswith("layer_") or "/layer_" in k

    def kto_loss_fn(lora, base_p, tokens, labels, kl_tokens, kl_labels,
                    patches, pos_xy, is_des):
        # 非 LLM(视觉/embedder)lora 叶必须 stop_gradient —— zeros 仍可导,
        # 不切断则视觉塔反向 ~32G ×2 前向 = 62G OOM(电池实测)
        lora_h = jax.tree_util.tree_map_with_path(
            lambda p, x: x.astype(jnp.bfloat16) if _is_llm(p)
            else jax.lax.stop_gradient(x.astype(jnp.bfloat16)), lora)
        pol = gpeft.merge_params(base_p, lora_h)
        pvi = PreprocessedVisionInput(
            patches=patches, positions_xy=pos_xy, soft_token_counts=counts)
        lp_pol = sum_logprob_fn(pol, tokens, labels, pvi)
        kl_pol = sum_logprob_fn(pol, kl_tokens, kl_labels, pvi)
        ref = gpeft.merge_params(
            base_p, jax.tree.map(jnp.zeros_like, lora_h))   # ref ≡ base
        lp_ref = jax.lax.stop_gradient(
            sum_logprob_fn(ref, tokens, labels, pvi))
        kl_ref = jax.lax.stop_gradient(
            sum_logprob_fn(ref, kl_tokens, kl_labels, pvi))
        kl = jnp.clip((kl_pol - kl_ref).mean(), 0.0)
        kl = jax.lax.stop_gradient(kl)
        logratio = lp_pol - lp_ref
        loss = jnp.where(
            is_des > 0,
            a.w_desirable * (1 - jax.nn.sigmoid(a.beta * (logratio - kl))),
            a.w_undesirable * (1 - jax.nn.sigmoid(a.beta * (kl - logratio))))
        return loss.mean(), (kl, jnp.abs(logratio).mean())

    import functools
    from jax.sharding import Mesh, PartitionSpec as P
    from jax.experimental.shard_map import shard_map

    devs = jax.devices()
    DP = a.dp or len(devs)
    mesh = Mesh(np.asarray(devs[:DP]), ("dp",))
    print(f"[kto] dp={DP}(每设备 1 对/micro,错配 KL 为 batch 维滚动)")

    def grad_local(lora, *b):
        (loss, m), grads = jax.value_and_grad(
            kto_loss_fn, has_aux=True)(lora, *b)
        pm = lambda x: jax.lax.pmean(x, "dp")
        return pm(loss), tuple(pm(x) for x in m), jax.tree.map(pm, grads)

    grad_sharded = shard_map(
        grad_local, mesh=mesh,
        in_specs=(P(), P()) + (P("dp"),) * 7,   # base 显式复制进 Manual ctx
        out_specs=(P(), (P(), P()), P()), check_rep=False)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(lora, opt_state, *b):
        loss, m, grads = grad_sharded(lora, base, *b)
        updates, opt_state = optim.update(grads, opt_state, lora)
        return optax.apply_updates(lora, updates), opt_state, loss, m

    lora = lora0
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    n_micro = a.steps * a.accum
    cursor = 0
    for micro in range(n_micro):
        built = [build(pairs[(cursor + j) % len(pairs)]) for j in range(DP)]
        cursor += DP
        toks = np.stack([b[0] for b in built])       # [DP, L]
        labs = np.stack([b[1] for b in built])
        pt = np.stack([b[2] for b in built])
        px = np.stack([b[3] for b in built])
        is_des = np.asarray([b[4] for b in built], np.float32)
        # 错配 KL: batch 维滚动 —— (video_i, completion_{i-1 mod DP})
        roll = np.roll(np.arange(DP), 1)
        kl_toks = np.concatenate([toks[:, :T], toks[roll][:, T:]], axis=1)
        kl_labs = np.concatenate([labs[:, :T], labs[roll][:, T:]], axis=1)
        lora, opt_state, loss, (kl, lr_abs) = step(
            lora, opt_state,
            jnp.asarray(toks), jnp.asarray(labs),
            jnp.asarray(kl_toks), jnp.asarray(kl_labs),
            jnp.asarray(pt), jnp.asarray(px), jnp.asarray(is_des))
        if micro == 0:
            print(f"[compile+step0] {time.time()-t0:.0f}s", flush=True)
            t0 = time.time()
        if (micro + 1) % a.accum == 0:
            o = (micro + 1) // a.accum
            print(f"[kto] step {o}/{a.steps} loss={float(loss):.4f} "
                  f"kl={float(kl):.3f} |logratio|={float(lr_abs):.3f} "
                  f"micro_s/it={(time.time()-t0)/max(micro,1):.2f}",
                  flush=True)
    flat = jax.tree_util.tree_flatten_with_path(lora)[0]
    import numpy as _np
    _np.savez(os.path.join(a.out, "lora_params.npz"),
              **{"lora/" + "/".join(getattr(k, "key", str(k)) for k in p):
                 _np.asarray(v) for p, v in flat})
    print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
