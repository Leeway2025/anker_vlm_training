"""Gate B(基准侧): 用现有 torch/HF 栈导出 16 帧 + 生产 prompt 的 token 排布。

这是 Gate B 的"真值"——RKLLM 端侧按 HF 语义运行,JAX 侧必须与它对齐。
用现有 torch venv 在仓库根目录运行(唯一允许 import 现有代码的 PoC):
  <torch_venv>/bin/python jax_impl/poc/02a_dump_hf_layout.py --out /tmp/hf_layout.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def blocks_of(mask):
    """mm_token_type_ids 里连续 =1 段 → [(start, len), ...](每帧一个块)。"""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - s))
            s = None
    if s is not None:
        out.append((s, len(mask) - s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/hf_layout.json")
    a = ap.parse_args()

    import numpy as np
    import yaml
    from transformers import AutoProcessor
    from data.build_dataset import AnkerCollator

    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(cfg["model"]["name_or_path"])
    col = AnkerCollator(processor, cfg)

    nf = cfg["sampling"]["num_frames"]
    sz = cfg["sampling"]["image_size"]
    # 固定伪帧(全 128 灰),两侧像素一致 → 排布对齐与像素无关
    frames = np.full((nf, sz, sz, 3), 128, np.uint8)
    enc = col._encode_prompt(frames, "")

    ids = enc["input_ids"][0].tolist()
    mm = enc["mm_token_type_ids"][0].tolist()
    blks = blocks_of(mm)
    tok = processor.tokenizer
    layout = {
        "num_frames": nf, "image_size": sz,
        "total_len": len(ids),
        "soft_token_total": sum(mm),
        "vision_blocks": blks,                      # 每帧 (起点, 长度)
        "tokens_per_frame": sorted({b[1] for b in blks}),
        "input_ids": ids,
        "mm_token_type_ids": mm,
        # 每个视觉块前后 3 个 token 的可读形式(对齐特殊 token 用)
        "block_context": [
            {"before": tok.convert_ids_to_tokens(ids[max(0, s - 3):s]),
             "after": tok.convert_ids_to_tokens(ids[s + l:s + l + 3])}
            for s, l in blks[:3]],
        "text_head": tok.convert_ids_to_tokens(ids[:16]),
    }
    json.dump(layout, open(a.out, "w"))
    print(f"[OK] total={layout['total_len']} soft={layout['soft_token_total']} "
          f"frames={len(blks)} tokens/frame={layout['tokens_per_frame']}")
    print(f"[OK] 基准排布 -> {a.out}")


if __name__ == "__main__":
    main()
