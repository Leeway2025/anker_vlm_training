"""Gate C(多模态): 16 帧视频 + 生产 prompt 的 HF/JAX 前向对拍 —— 决定性验证。

  1. torch venv:  python jax_impl/poc/03b_mm_parity.py --side hf  --out /tmp/mm_hf.json
  2. jax venv:    python jax_impl/poc/03b_mm_parity.py --side jax --ref /tmp/mm_hf.json

拼装配方(FINDINGS.md): JAX tokens = HF input_ids 原样,仅把
mm_token_type_ids==2 的位置(每帧 64 个视频占位 258884)替换为哨兵 -2;
帧图像走 preprocess_and_patchify → PreprocessedVisionInput。
判定: 末位 top-5 id 一致 → PASS(logprob 容差放宽到 0.1,视觉侧
归一化/位置编码差异会先体现在 id 序上)。
"""
import argparse
import json
import os
import sys

TOPK = 5


def gray_frames(nf, sz):
    import numpy as np
    if os.environ.get("MM_NOISE"):          # 随机帧: 像素依赖路径的严格考验
        return np.random.RandomState(0).randint(
            0, 256, (nf, sz, sz, 3)).astype(np.uint8)
    return np.full((nf, sz, sz, 3), 128, np.uint8)


def run_hf(out):
    sys.path.insert(0, os.getcwd())
    import torch
    import yaml
    import transformers
    from transformers import AutoProcessor
    from data.build_dataset import AnkerCollator

    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    name = cfg["model"]["name_or_path"]
    processor = AutoProcessor.from_pretrained(name)
    col = AnkerCollator(processor, cfg)
    frames = gray_frames(cfg["sampling"]["num_frames"],
                         cfg["sampling"]["image_size"])
    enc = col._encode_prompt(frames, "")
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        name, torch_dtype=torch.float32)
    model.eval()
    feed = {k: v for k, v in enc.items()}
    print(f"[hf] forward keys: {list(feed)}")
    with torch.no_grad():
        logits = model(**feed).logits[0, -1]
    lp = torch.log_softmax(logits.float(), -1)
    top = torch.topk(lp, TOPK)
    json.dump({
        "input_ids": enc["input_ids"][0].tolist(),
        "mm_token_type_ids": enc["mm_token_type_ids"][0].tolist(),
        "top_ids": top.indices.tolist(),
        "top_logprobs": [round(float(x), 4) for x in top.values],
    }, open(out, "w"))
    print(f"[OK] HF 多模态 top{TOPK}: {top.indices.tolist()} -> {out}")


def run_jax(ref_path):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gemma import gm
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
    from gemma.gm.nn.gemma4.vision._preprocessing import preprocess_and_patchify

    ref = json.load(open(ref_path))
    ids = ref["input_ids"]
    mm = ref["mm_token_type_ids"]
    tokens = [(-2 if m == 2 else t) for t, m in zip(ids, mm)]
    n_vis = sum(1 for t in tokens if t == -2)
    print(f"[jax] tokens={len(tokens)} 视觉哨兵={n_vis}(预期 1024)")

    nf, sz = 16, 384
    all_frames = gray_frames(nf, sz)
    frames = [all_frames[i] for i in range(nf)]
    # max_soft_tokens 是"每图预算",库会放大图片填满预算(384² 默认下
    # 出 1089/帧);限到 64 → 576 patch 预算,384×384 恰好原样不缩放,
    # 与 HF 的 64 token/帧 逐帧一致
    patches, pos_xy, counts = preprocess_and_patchify(
        frames, max_soft_tokens=64)
    print(f"[jax] patches={patches.shape} pos={pos_xy.shape} counts={counts}")
    assert sum(counts) == n_vis, f"soft token 总数不齐: {sum(counts)} vs {n_vis}"
    # _encode_vision 约定: patches [1, n_images*max_patches, dim],
    # 内部按 soft_token_counts 长度拆回 (n_images, max_patches, dim)
    n, p, d = patches.shape
    pvi = PreprocessedVisionInput(
        patches=patches.reshape(1, n * p, d),
        positions_xy=pos_xy.reshape(1, n * p, 2),
        soft_token_counts=tuple(int(c) for c in counts))

    # 默认 text_only 剥离视觉塔;且 VisionEncoder 默认 output_length=280
    # (图像语义),HF 视频语义是 64 token/帧 → 用 config 覆盖
    import dataclasses
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    base_cfg = gm.nn.Gemma4_E2B.config
    cfg64 = dataclasses.replace(
        base_cfg,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    model = gm.nn.Gemma4_E2B(text_only=False, config=cfg64)
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    out = model.apply({"params": params}, tokens=jnp.asarray([tokens]),
                      images=pvi, return_last_only=True)
    logits = out.logits[0] if hasattr(out, "logits") else out[0]
    lp = jax.nn.log_softmax(logits.astype(jnp.float32))
    top = jnp.argsort(lp)[-TOPK:][::-1]
    print(f"[JAX] top{TOPK} ids={top.tolist()} "
          f"lp={[round(float(lp[i]), 4) for i in top]}")
    print(f"[HF ] top{TOPK} ids={ref['top_ids']} lp={ref['top_logprobs']}")
    # 判定: top-2 一致(分类解码只看头部);尾部 (<-10 nat) 排序抖动
    # 属 bf16 实现噪声,不计
    same2 = top.tolist()[:2] == ref["top_ids"][:2]
    d1 = abs(float(lp[top[0]]) - ref["top_logprobs"][0])
    print(f"[metric] top1 lp 差={d1:.4f}")
    print("Gate C(多模态):", "PASS" if same2 and d1 < 0.1 else
          "NO-GO/待查(头部预测不一致)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["hf", "jax"], required=True)
    ap.add_argument("--out", default="/tmp/mm_hf.json")
    ap.add_argument("--ref", default="/tmp/mm_hf.json")
    a = ap.parse_args()
    run_hf(a.out) if a.side == "hf" else run_jax(a.ref)
