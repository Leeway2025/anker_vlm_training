"""阶段① 重采样器热身(独立小循环;只训 tok_resampler_*,其余一切冻结)。

目标: 重采样器输出回归『dyn 全局竞争选出的 512 token』——
  teacher = _compress_soft_tokens(t, K, "dyn")(同一 batch 的 soft token 上
  现算,stop_gradient,无需外部标签);loss = MSE(student, teacher)。
动机: 让 512 个 query 先学会"近似复现现有 dyn 选择的内容与光栅序",把
  重采样器从随机初始化拉到一个与现管线兼容的起点 —— 阶段②(train_sft
  TOKEN_RESAMPLER=1 + --init-npz 本产物)再端到端换真任务 loss。

为什么独立小循环而不复用 train_sft(工程量对比,研究结论):
  - 本目标只需【视觉编码器前向(冻结)+ 重采样器】,5B LLM 完全不参与:
    单步成本比 train_sft 低一个数量级以上,CPU 亦可跑通冒烟;
  - 若复用 train_sft,预计改动点: loss_fn 加 MSE 分支(要在补丁外再拿到
    t 与 dyn teacher → 得把 _selected 改成可回传中间量或重复编码器前向)、
    optimizer 各组 lr=0 冻结、蒸馏互斥逻辑放宽 —— 改动面大且步耗高,不选。

产物: <out>/resampler_warmup.npz,键 "tok/<struct路径>"(与 train_sft 存档
  同款;struct 路径=顶层参数名,对齐 learnhead 产物 "tok/tok_scorer_A" 口径)
  → 阶段② train_sft --init-npz 直接热启(restore_train_tree 命中 tok/ 叶;
  npz 无 lora/ 键,LoRA 保留新初始化,不触发③号全零坑)。

用法(草稿,TPU 上跑;CPU 冒烟加 JAX_PLATFORMS=cpu --limit 16 --steps 2):
  TOKEN_RESAMPLER=1 python jax_impl/resampler_warmup.py \
      --labels /data/labels_100k_v2.jsonl --wds-dir <可选> \
      --steps 2000 --bs 8 --lr 1e-4 --k 32 --out outputs/resampler_warmup
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--wds-dir", default=None,
                    help="显式指定分片目录(覆盖 labels 内 meta.wds_dir)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=32,
                    help="dyn teacher 的每帧预算 K(n*K 须 == RESAMPLER_TOKENS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条(冒烟)")
    ap.add_argument("--val-n", type=int, default=64, help="尾部 N 条当验证集")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out", default="outputs/resampler_warmup")
    a = ap.parse_args()
    from jax_impl.logtee import tee_stdio
    tee_stdio(a.out)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from gemma import gm
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from jax_impl.data import (load_frames, make_vision_input,
                               _compress_soft_tokens, _resample_soft_tokens,
                               _resampler_specs, resampler_init_leaf)

    mst = int(os.environ.get("MAX_SOFT_TOKENS", "64"))
    total = int(os.environ.get("RESAMPLER_TOKENS", "512"))
    layers = int(os.environ.get("RESAMPLER_LAYERS", "1"))
    heads = int(os.environ.get("RESAMPLER_HEADS", "8"))
    ffn = os.environ.get("RESAMPLER_FFN", "1") == "1"
    ffn_mult = int(os.environ.get("RESAMPLER_FFN_MULT", "2"))

    # ---- 数据: 只要帧,不建 token/不要 tokenizer(与 SftDataset 解耦)----
    recs = [json.loads(l) for l in open(a.labels, encoding="utf-8")]
    if a.limit:
        recs = recs[: a.limit]
    val_recs = recs[-a.val_n:] if a.val_n else []
    tr_recs = recs[: len(recs) - len(val_recs)]
    print(f"[data] train={len(tr_recs)} val={len(val_recs)} from {a.labels}")

    def _frames(rec):
        if a.wds_dir:
            rec = {**rec, "meta": {**(rec.get("meta") or {}),
                                   "wds_dir": a.wds_dir}}
        return load_frames(rec, a.wds_dir or os.path.dirname(a.labels))

    # ---- 视觉编码器(冻结,bf16): 只取 base 的 vision_encoder 子树 ----
    ve = gemma_vision.VisionEncoder(use_clipped_linears=True,
                                    output_length=mst)
    base = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    ve_params = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16),
                             base["vision_encoder"])
    D = ve.d_model
    del base                                    # 立即释放 5B 其余子树
    print(f"[ve] 视觉编码器就绪 D={D} mst={mst}(冻结,bf16)")

    # ---- 重采样器参数(唯一可训树;初始化与 train_sft 同源)----
    rng = np.random.RandomState(a.seed)
    prm0 = {name: jnp.asarray(resampler_init_leaf(name, shape, rng))
            for name, shape, _k in _resampler_specs(
                D, total, layers, ffn, ffn_mult)}
    n_par = sum(int(v.size) for v in prm0.values())
    print(f"[rsp] 可训参数 {len(prm0)} 叶 / {n_par/1e6:.2f}M "
          f"(TOTAL={total} L={layers} H={heads} FFN={int(ffn)})")

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    opt_state = tx.init(prm0)

    def _encode(pa, px):
        """[B, n*p, d] patch → t[B, n, mst, D] fp32(编码器冻结 stop_grad)。"""
        B = pa.shape[0]
        n = pa.shape[1] // (mst * 9)             # 每帧 9*mst 个 patch
        p = pa.shape[1] // n
        emb, _m = ve.apply({"params": ve_params},
                           jnp.reshape(pa, (B * n, p, pa.shape[2])),
                           jnp.reshape(px, (B * n, p, 2)))[0]
        t = jnp.reshape(emb[:, :mst, :], (B, n, mst, -1)).astype(jnp.float32)
        return jax.lax.stop_gradient(t)

    def loss_fn(prm, pa, px):
        t = _encode(pa, px)
        n = t.shape[1]
        assert n * a.k == total, (
            f"dyn teacher n*K={n * a.k} != RESAMPLER_TOKENS={total}")
        # teacher = 现有 dyn 全局竞争选择(floor8+竞争,保光栅序)——
        # 与生产 dyn K=32 口径完全一致;stop_grad 只当回归靶
        teacher = jax.lax.stop_gradient(
            _compress_soft_tokens(t, a.k, "dyn"))
        student = _resample_soft_tokens(t, prm, heads)
        mse = jnp.mean((student - teacher) ** 2)
        # rel = mse / 靶二阶矩:尺度无关的进度读数(soft token 幅值极大,
        # 裸 MSE 数字吓人但无信息;rel<1 才说明比"输出全零"强)
        return mse, mse / (jnp.mean(teacher ** 2) + 1e-9)

    @jax.jit
    def step(prm, opt_state, pa, px):
        (loss, rel), g = jax.value_and_grad(loss_fn, has_aux=True)(
            prm, pa, px)
        up, opt_state = tx.update(g, opt_state)
        return optax.apply_updates(prm, up), opt_state, loss, rel

    eval_j = jax.jit(loss_fn)

    def batch(rs):
        pa, px, _c = make_vision_input([_frames(r) for r in rs])
        return jnp.asarray(pa), jnp.asarray(px)

    os.makedirs(a.out, exist_ok=True)
    prm, best = prm0, (1e9, -1)
    order = rng.permutation(len(tr_recs))
    cur, t0 = 0, time.time()
    for s in range(1, a.steps + 1):
        if cur + a.bs > len(tr_recs):            # 简易 epoch 重洗
            order = rng.permutation(len(tr_recs))
            cur = 0
        rs = [tr_recs[i] for i in order[cur: cur + a.bs]]
        cur += a.bs
        prm, opt_state, loss, rel = step(prm, opt_state, *batch(rs))
        if s == 1 or s % 20 == 0:
            print(f"[warmup] step {s}/{a.steps} mse={float(loss):.4e} "
                  f"rel={float(rel):.4f} ({time.time()-t0:.0f}s)", flush=True)
        if val_recs and s % a.eval_every == 0:
            vl = [float(eval_j(prm, *batch(val_recs[i: i + a.bs]))[0])
                  for i in range(0, len(val_recs) - a.bs + 1, a.bs)]
            v = sum(vl) / max(len(vl), 1)
            tag = ""
            if v < best[0]:
                best, tag = (v, s), " *best"
                np.savez(os.path.join(a.out, "resampler_warmup.npz"),
                         **{"tok/" + k: np.asarray(x, np.float32)
                            for k, x in prm.items()})
            print(f"[eval] step {s} val_mse={v:.4e}{tag}", flush=True)
    if not os.path.exists(os.path.join(a.out, "resampler_warmup.npz")):
        np.savez(os.path.join(a.out, "resampler_warmup.npz"),
                 **{"tok/" + k: np.asarray(x, np.float32)
                    for k, x in prm.items()})
    json.dump({"best_val_mse": best[0], "best_step": best[1],
               "total": total, "layers": layers, "heads": heads,
               "ffn": ffn, "ffn_mult": ffn_mult, "k": a.k, "D": int(D)},
              open(os.path.join(a.out, "warmup_meta.json"), "w"))
    print(f"[save] {a.out}/resampler_warmup.npz "
          f"(best val_mse={best[0]:.4e}@{best[1]})——阶段② train_sft "
          f"TOKEN_RESAMPLER=1 --init-npz 本产物热启")


if __name__ == "__main__":
    main()
