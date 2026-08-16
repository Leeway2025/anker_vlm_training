"""老师梯度显著性(teacher gradient saliency)抽取器。

对每条样本 teacher-forcing 正确答案,取『正确 RT 字母 logit + 正确 SubKS 字母
logit』之和,对每个池化 soft token 求显著性 = ∂loss/∂w_i(w=每 token 乘性权重,
暖启=1)= Σ_D (∂loss/∂t_iD)·t_iD = gradient×input。不做压缩(全 64/帧=1024 token
送 LLM,层布 hf_layout 恰 1024 哨兵),故显著性覆盖全部 64/帧,可据此选 K=32。

关键:w 作为 params 叶(经 model.apply 正常穿过内层 jit)→ 梯度可回流且可 jit
(全程 global 注入会被内层 jit 当常量截断梯度=0,故弃用)。

输出:
  <out>.npz   gi[M,n,64]=|∂loss/∂w|(gradient×input 绝对值), sg[M,n,64]=带符号
  <out>.ids.json  video_id 列表

  python jax_impl/attrib_saliency.py --labels /data/labels_test.jsonl \
      --layout /data/hf_layout.json --init-npz <teacher.npz> \
      --rank-scheme prod --limit 200 --shard 0/8 --out outputs/sal/test
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--wds-dir", default=None)
    ap.add_argument("--init-npz")
    ap.add_argument("--rank-scheme",
                    choices=["auto", "uniform", "prod", "map"], default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-proj", action="store_true")
    ap.add_argument("--max-new", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from jax_impl.logtee import tee_stdio
    tee_stdio(os.path.dirname(a.out) or ".", name=os.path.basename(a.out) + ".sal.log")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from gemma import gm
    from gemma import peft as gpeft
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
    from jax_impl.data import SftDataset, make_vision_input
    from flax.traverse_util import flatten_dict, unflatten_dict

    g4_tr._token_utils.remove_mm_logits = \
        lambda logits, tokens, num_tokens_per_image: logits

    NFR = 16      # 帧数(hf_layout=16 帧×64)
    CNT = 64

    # ---- 注册 tok_w[NFR,CNT] 乘性权重(setup 补丁,暖启=1),并改写 _encode_vision
    #      为『全 64/帧 × tok_w』,梯度目标就是 tok_w ----
    _orig_setup = g4_tr.Transformer.setup

    def _setup_w(self):
        _orig_setup(self)
        if self.config.vision_encoder is None:
            return
        self.tok_w = self.param(
            "tok_w", lambda key, shape, dtype=jnp.float32: jnp.ones(
                shape, jnp.float32), (NFR, CNT))
    g4_tr.Transformer.setup = _setup_w

    def _encode_w(self, vision_input):
        patches = vision_input.patches
        counts = vision_input.soft_token_counts
        n = len(counts)
        B = patches.shape[0]
        p, d = patches.shape[1] // n, patches.shape[2]
        pa = jnp.reshape(patches, (B * n, p, d))
        px = jnp.reshape(vision_input.positions_xy, (B * n, p, 2))
        emb, _m = self.vision_encoder(pa, px)[0]
        t = jnp.reshape(emb[:, :CNT, :], (B, n, CNT, -1))
        w = self.tok_w[None, :, :, None].astype(t.dtype)   # [1,n,64,1]
        t = t * w
        sel = jnp.reshape(t, (B, n * CNT, t.shape[-1]))
        return self.embedder.encode_vision(sel[:, None])[:, 0]
    g4_tr.Transformer._encode_vision = _encode_w

    # ---- rank 方案 ----
    z, uni_rank, has_lora = None, 16, False
    if a.init_npz:
        from jax_impl.npz_io import (detect_rank_scheme, load_lora_strict,
                                     merge_proj_into_base)
        z = np.load(a.init_npz)
        has_lora = any(k.startswith("lora/") for k in z.files)
        if has_lora:
            det_scheme, ranks = detect_rank_scheme(z)
            scheme = a.rank_scheme if a.rank_scheme != "auto" else det_scheme
            print(f"[scheme] {scheme}")
            if scheme == "prod":
                from jax_impl.prod_lora import install_prod_lora
                install_prod_lora()
            elif scheme == "map":
                from jax_impl.prod_lora import install_map_lora
                install_map_lora(ranks)
            else:
                uni_rank = ranks[0]

    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    tok = gm.text.Gemma4Tokenizer()
    ds = SftDataset(a.labels, a.layout, tok, wds_dir=a.wds_dir,
                    max_label_len=a.max_new)
    T = len(ds.template)
    L = T + a.max_new
    print(f"[sal] T={T} L={L}")

    if a.init_npz and has_lora:
        model = gm.nn.LoRA(rank=uni_rank,
                           model=gm.nn.Gemma4_E2B(text_only=False, config=cfg64))
    else:
        model = gm.nn.Gemma4_E2B(text_only=False, config=cfg64)
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    params = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), params)
    if a.init_npz:
        params, n_proj = merge_proj_into_base(
            params, z, jnp, jnp.bfloat16, required=not a.no_proj)

    ex0 = ds[0]
    p0, x0, counts0 = make_vision_input([ex0["frames"]])
    _pvi0 = PreprocessedVisionInput(
        patches=jnp.asarray(p0), positions_xy=jnp.asarray(x0),
        soft_token_counts=counts0)
    struct = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0), tokens=jnp.zeros((1, L), jnp.int32),
        images=_pvi0))
    if a.init_npz and has_lora:
        lora_struct = gpeft.split_params(struct["params"])[1]
        lora = load_lora_strict(z, lora_struct, jnp, jnp.bfloat16)
        params = gpeft.merge_params(params, lora)
        print(f"[init] lora+proj({n_proj}叶) from {a.init_npz}")

    # 定位 tok_w 叶路径,注入 ones 到 params
    w_key = None
    for path, _leaf in jax.tree_util.tree_flatten_with_path(struct["params"])[0]:
        keys = tuple(getattr(kk, "key", str(kk)) for kk in path)
        if keys[-1] == "tok_w":
            w_key = keys
            break
    assert w_key is not None, "tok_w 未注册?"
    fp = flatten_dict(params)
    fp[w_key] = jnp.ones((NFR, CNT), jnp.float32)
    w0 = fp[w_key]
    rest = {k: v for k, v in fp.items() if k != w_key}
    print(f"[sal] tok_w @ {'/'.join(w_key)}")

    def loss_fn(w, rest_flat, tokens, pvi, rt_id, sk_id):
        fp2 = dict(rest_flat)
        fp2[w_key] = w
        par = unflatten_dict(fp2)
        out = model.apply({"params": par}, tokens=tokens, images=pvi)
        lg = out.logits[0].astype(jnp.float32)
        return lg[T - 1, rt_id] + lg[T + 1, sk_id]

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

    def _mk_pvi(p, x, counts):
        return PreprocessedVisionInput(
            patches=jnp.asarray(p), positions_xy=jnp.asarray(x),
            soft_token_counts=counts)

    recs = ds.recs
    if a.shard:
        i, nn = (int(x) for x in a.shard.split("/"))
        recs = recs[i::nn]
    if a.limit:
        recs = recs[: a.limit]
    id2idx = {r["video_id"]: k for k, r in enumerate(ds.recs)}

    import time
    sgs, ids = [], []
    t0 = time.time()
    for j, r in enumerate(recs):
        ex = ds[id2idx[r["video_id"]]]
        toks = jnp.asarray(ex["tokens"][None])
        p, x, counts = make_vision_input([ex["frames"]])
        pvi = _mk_pvi(p, x, counts)
        rt_id = int(ex["tokens"][T])
        sk_id = int(ex["tokens"][T + 2])
        g = grad_fn(w0, rest, toks, pvi, jnp.int32(rt_id), jnp.int32(sk_id))
        sg = np.asarray(g, np.float32)                 # [n,64] signed grad·input
        sgs.append(sg)
        ids.append(r["video_id"])
        if j < 3 or (j + 1) % 50 == 0:
            dt = time.time() - t0
            ga = np.abs(sg)
            print(f"[sal] {j+1}/{len(recs)} {r['video_id']} "
                  f"|gi|[max={ga.max():.3e} mean={ga.mean():.3e} "
                  f"spread={ga.std():.3e}] ({dt:.1f}s)")
    sg = np.stack(sgs)
    np.savez(a.out + ".npz", sg=sg, gi=np.abs(sg))
    json.dump(ids, open(a.out + ".ids.json", "w"))
    print(f"[sal] 完成 M={len(ids)} → {a.out}.npz 用时 {time.time()-t0:.1f}s "
          f"({(time.time()-t0)/max(1,len(ids)):.1f}s/条)")


if __name__ == "__main__":
    main()
