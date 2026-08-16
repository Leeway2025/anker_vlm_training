"""老师梯度显著性提取(0814 学习打分头流水线①)。

对每个训练样本:老师(满秩汤,SELECT_TOKENS_K 关闭=全 1024 token)前向
教师强制序列(template+gold),对「两个分类字母位的 gold logit 之和」反传
到 patches,|grad| 的 D 维范数 → [16,576] → 3×3 组内求和 → [16,64]
显著性图(每片段归一化到 0-255 uint8)。产物供打分头(MLP)回归训练。

  python3 jax_impl/extract_saliency.py --init-npz outputs/soupw1/soupw1.npz \
      --labels /data/labels_train_plus_testval_v2.jsonl \
      --layout /data/hf_layout.json --n 100000 --shard 0/8 \
      --out outputs/saliency/sal_shard0.npz
"""
import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-npz", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--wds-dir")
    ap.add_argument("--n", type=int, default=100000, help="取前 N 个 train 样本")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))

    assert not int(os.environ.get("SELECT_TOKENS_K", "0")), \
        "显著性须在全 token 输入下提取,请勿设 SELECT_TOKENS_K"

    import jax
    import jax.numpy as jnp
    import numpy as np
    from gemma import gm
    from gemma import peft as gpeft
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
    from jax_impl.data import SftDataset, make_vision_input
    from jax_impl.npz_io import (detect_rank_scheme, load_lora_strict,
                                 merge_proj_into_base)

    # 与 infer.py 同款:HF 语义对齐补丁(勿删)
    g4_tr._token_utils.remove_mm_logits = \
        lambda logits, tokens, num_tokens_per_image: logits

    # 反传必须 remat(单芯 HBM 扛不住 847 token 全激活;照搬 train_sft/grpo)
    from flax import linen as nn
    from gemma.gm.nn.gemma4 import _modules as g4_modules
    from gemma.gm.nn.gemma4.vision import _transformer as gv_tr
    if not getattr(g4_modules, "_REMAT_PATCHED", False):
        _orig_call = g4_modules.Block.__call__
        _POL = jax.checkpoint_policies.nothing_saveable

        def _make_core(skip_flag):
            def core(self, xx, pos, cache, mask, pli, kvs):
                return _orig_call(self, xx, pos, cache, mask, pli, kvs,
                                  skip_sliding_mask=skip_flag)
            return nn.remat(core, policy=_POL, prevent_cse=False)

        _core = {False: _make_core(False), True: _make_core(True)}

        def patched(self, xx, segment_pos, cache, attn_mask,
                    per_layer_input=None, kv_shared_cache=None,
                    skip_sliding_mask=False):
            return _core[bool(skip_sliding_mask)](
                self, xx, segment_pos, cache, attn_mask,
                per_layer_input, kv_shared_cache)

        g4_modules.Block.__call__ = patched
        g4_modules._REMAT_PATCHED = True
    if not getattr(gv_tr, "_REMAT_PATCHED", False):
        gv_tr.VisionBlock = nn.remat(
            gv_tr.VisionBlock,
            policy=jax.checkpoint_policies.nothing_saveable)
        gv_tr._REMAT_PATCHED = True
        print("[remat] LLM Block + VisionBlock 已包装")

    z = np.load(a.init_npz)
    det_scheme, ranks = detect_rank_scheme(z)
    if det_scheme == "prod":
        from jax_impl.prod_lora import install_prod_lora
        install_prod_lora()
    elif det_scheme == "map":
        from jax_impl.prod_lora import install_map_lora
        install_map_lora(ranks)
    uni_rank = 16 if isinstance(ranks, dict) else ranks[0]

    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    tok = gm.text.Gemma4Tokenizer()
    ds = SftDataset(a.labels, a.layout, tok, wds_dir=a.wds_dir)
    model = gm.nn.LoRA(rank=uni_rank,
                       model=gm.nn.Gemma4_E2B(text_only=False, config=cfg64))
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    params = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), params)
    params, n_proj = merge_proj_into_base(params, z, jnp, jnp.bfloat16,
                                          required=False)
    ex0 = ds[ds.train_idx[0]]
    p0, x0, counts = make_vision_input([ex0["frames"]])
    pvi0 = PreprocessedVisionInput(
        patches=jnp.asarray(p0), positions_xy=jnp.asarray(x0),
        soft_token_counts=counts)
    struct = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0),
        tokens=jnp.zeros((1, len(ex0["tokens"])), jnp.int32), images=pvi0))
    lora = load_lora_strict(z, gpeft.split_params(struct["params"])[1],
                            jnp, jnp.bfloat16)
    params = gpeft.merge_params(params, lora)
    print(f"[init] lora+proj({n_proj}) from {a.init_npz} scheme={det_scheme}")

    def gold_logit_sum(patches, pos_xy, tokens, letter_pos):
        pvi = PreprocessedVisionInput(
            patches=patches, positions_xy=pos_xy, soft_token_counts=counts)
        out = model.apply({"params": params}, tokens=tokens, images=pvi)
        lg = (out.logits if hasattr(out, "logits") else out)[0]
        # 教师强制: 位置 p 的 token 由 logits[p-1] 预测
        s = 0.0
        for j in range(2):
            p = letter_pos[j]
            s = s + lg[p - 1, tokens[0, p]].astype(jnp.float32)
        return s

    grad_fn = jax.jit(jax.grad(gold_logit_sum, argnums=0))

    idxs = [i for k, i in enumerate(ds.train_idx[:a.n]) if k % sn == si]
    print(f"[saliency] shard {si}/{sn}: {len(idxs)} 样本")
    out_maps, out_ids = [], []
    import time
    t0 = time.time()
    for k, i in enumerate(idxs):
        ex = ds[i]
        w = ex["weights"]
        lp = np.where(w == np.max(w[w > 0]))[0][:2]   # 两个字母位(cls_w 最大)
        if len(lp) < 2:
            continue
        p, x, _ = make_vision_input([ex["frames"]])
        g = grad_fn(jnp.asarray(p), jnp.asarray(x),
                    jnp.asarray(ex["tokens"][None]),
                    tuple(int(v) for v in lp))
        sal = np.asarray(jnp.linalg.norm(
            g[0].astype(jnp.float32), axis=-1))       # [n_patch_total]
        # 按 positions 分箱到 8×8 token 网格(帧可非正方形,勿假设 24×24):
        pos = np.asarray(x)[0].astype(np.float64)     # [n_patch_total, 2]
        P = pos.shape[0] // 16
        sal16 = np.zeros((16, 64), np.float64)
        for f in range(16):
            pp = pos[f * P:(f + 1) * P]
            ss = sal[f * P:(f + 1) * P]
            r0, r1 = pp[:, 0].min(), pp[:, 0].max()
            c0, c1 = pp[:, 1].min(), pp[:, 1].max()
            rb = np.clip(((pp[:, 0] - r0) / (r1 - r0 + 1e-9) * 8).astype(int),
                         0, 7)
            cb = np.clip(((pp[:, 1] - c0) / (c1 - c0 + 1e-9) * 8).astype(int),
                         0, 7)
            np.add.at(sal16[f], rb * 8 + cb, ss)
        sal = sal16
        m = sal.max()
        sal8 = (sal / (m + 1e-9) * 255.0).astype(np.uint8)
        out_maps.append(sal8)
        out_ids.append(ex["video_id"])
        if (k + 1) % 200 == 0:
            dt = time.time() - t0
            print(f"[saliency] {k+1}/{len(idxs)} {dt/(k+1):.2f}s/样本", flush=True)
            np.savez(a.out, ids=np.array(out_ids),
                     maps=np.stack(out_maps))          # 断点友好: 周期覆盖写
    np.savez(a.out, ids=np.array(out_ids), maps=np.stack(out_maps))
    print(f"[saliency] 完成 {len(out_ids)} → {a.out}")


if __name__ == "__main__":
    main()
