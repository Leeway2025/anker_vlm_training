"""批量视觉等价测试: batch=2 前向 ≡ 两次 bs=1 前向(逐位对拍)。

  JAX_PLATFORMS=cpu python jax_impl/poc/05_batch_parity.py

背景(源码结论):
  - merge_flat_embeddings 对 batch vmap,需视觉嵌入 [B,T,D] ✅ 天然支持;
  - _encode_vision 写死 B=1(reshape 忽略 batch 维,返回 [1,T,D])❌。
本测试内联"batch 化 _encode_vision"补丁(与 train_sft 同一份逻辑):
  B=1 走原路径;B>1 → [B×n, p, d] 批量过 ViT → 折回 [B, n×cnt, D]。
判定: 两行样本的末位 logits 与 label 窗口 logits,batch 前向 vs 单样本
前向 max|Δ| < 1e-3(fp32)→ PASS,方可用于训练。
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def install_batched_encode_vision():
    """B>1 支持补丁;返回 True 表示已安装(幂等)。"""
    import jax.numpy as jnp
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    if getattr(g4_tr, "_BATCH_EV_PATCHED", False):
        return True
    _orig = g4_tr.Transformer._encode_vision

    def _batched(self, vision_input):
        patches = vision_input.patches
        B = patches.shape[0]
        if B == 1:
            return _orig(self, vision_input)
        counts = vision_input.soft_token_counts
        n = len(counts)
        cnt = counts[0]
        # 本任务恒 16 帧×64 token(均匀正方形帧,无 padding patch);
        # 非均匀场景需回退 bs=1 路径逐样本处理
        assert all(c == cnt for c in counts), "非均匀 soft_token_counts"
        p = patches.shape[1] // n
        d = patches.shape[2]
        pa = jnp.reshape(patches, (B * n, p, d))
        px = jnp.reshape(vision_input.positions_xy, (B * n, p, 2))
        emb, mask = self.vision_encoder(pa, px)[0]     # [B*n, l, D]
        toks = emb[:, :cnt, :]                          # 无 pad → 前 cnt 即真
        toks = jnp.reshape(toks, (B, n * cnt, toks.shape[-1]))
        toks = self.embedder.encode_vision(toks[:, None])[:, 0]
        return toks                                     # [B, n*cnt, D_model]

    g4_tr.Transformer._encode_vision = _batched
    g4_tr._BATCH_EV_PATCHED = True
    return True


def main():
    import dataclasses
    import numpy as np
    import jax
    import jax.numpy as jnp
    from gemma import gm
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
    from jax_impl.data import SftDataset, make_vision_input

    install_batched_encode_vision()
    # 训练语义: remove_mm_logits 旁路(v1 坑 7)
    g4_tr._token_utils.remove_mm_logits = \
        lambda logits, tokens, num_tokens_per_image: logits

    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    model = gm.nn.Gemma4_E2B(text_only=False, config=cfg64)
    tok = gm.text.Gemma4Tokenizer()
    ds = SftDataset("/dev/shm/fakedata/labels.jsonl",
                    "/dev/shm/hf_layout.json", tok)
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    params = jax.tree.map(lambda x: x.astype(jnp.float32), params)
    T = len(ds.template)

    exs = [ds[0], ds[1]]                    # 两个不同样本(不同帧不同 label)
    singles = []
    for e in exs:
        pt, px, counts = make_vision_input([e["frames"]])
        pvi = PreprocessedVisionInput(
            patches=jnp.asarray(pt), positions_xy=jnp.asarray(px),
            soft_token_counts=counts)
        out = model.apply({"params": params},
                          tokens=jnp.asarray(e["tokens"][None]), images=pvi)
        lg = out.logits if hasattr(out, "logits") else out
        singles.append(np.asarray(lg[0, T - 1:T + 8].astype(jnp.float32)))
        print(f"[bs1] {e['video_id']} 窗口 logits 就绪")

    pts, pxs = [], []
    for e in exs:
        pt, px, counts = make_vision_input([e["frames"]])
        pts.append(pt[0]); pxs.append(px[0])
    pvi2 = PreprocessedVisionInput(
        patches=jnp.asarray(np.stack(pts)),
        positions_xy=jnp.asarray(np.stack(pxs)),
        soft_token_counts=counts)
    toks2 = jnp.asarray(np.stack([e["tokens"] for e in exs]))
    out2 = model.apply({"params": params}, tokens=toks2, images=pvi2)
    lg2 = out2.logits if hasattr(out2, "logits") else out2
    ok = True
    for j in range(2):
        got = np.asarray(lg2[j, T - 1:T + 8].astype(jnp.float32))
        d = float(np.abs(got - singles[j]).max())
        print(f"[batch2] 行 {j} vs bs1: max|Δ|={d:.6f}")
        ok = ok and d < 1e-3
    print("批量视觉等价:", "PASS" if ok else "NO-GO")


if __name__ == "__main__":
    main()
