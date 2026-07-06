"""批量推理工具(hard_mining / build_kto_data / 评测共用)。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.sampling import decode_video, resize_production  # noqa: E402


def generate_predictions(model, processor, records, cfg, out_path,
                         max_new_tokens=64, batch_size=8):
    """对 records 逐批生成 "{RT} | {SubKS} | {desc}",写 preds jsonl。
    生产 prompt 原样;贪心解码(评测口径,无采样)。"""
    import torch
    prompt = open(cfg["format"]["prompt_file"], encoding="utf-8").read().strip()
    model.eval()
    done = set()
    if os.path.exists(out_path):                      # 断点续跑
        done = {json.loads(l)["video_id"] for l in open(out_path)}
    fout = open(out_path, "a", encoding="utf-8")

    batch, metas = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            inputs = processor(text=[prompt] * len(batch),
                               videos=batch, return_tensors="pt",
                               padding=True).to(model.device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
            texts = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)
        for vid, txt in zip(metas, texts):
            fout.write(json.dumps({"video_id": vid, "output": txt.strip()},
                                  ensure_ascii=False) + "\n")
        fout.flush()
        batch.clear()
        metas.clear()

    for rec in records:
        vid = rec["video_id"]
        if vid in done:
            continue
        path = os.path.join(cfg["data"]["video_root"],
                            os.path.basename(rec.get("video_uri", vid)))
        frames, _ = decode_video(path,
                                 num_frames=cfg["sampling"]["num_frames"])
        frames = resize_production(frames, cfg["sampling"]["image_size"])
        batch.append(list(frames))
        metas.append(vid)
        if len(batch) >= batch_size:
            flush()
    flush()
    fout.close()
    return out_path
