"""批量推理工具(hard_mining / build_kto_data / 评测共用)。

帧来源存储自适应(euno_wds.load_frames_for_record):
  meta.storage=='wds'(客户数据)→ 分片直读;否则视频文件解码。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.euno_wds import load_frames_for_record  # noqa: E402


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
            # do_sample_frames=False: 帧已按生产规则均匀采好,
            # 禁止 processor 内置采样器二次重采(默认 32 帧,烟测确认)
            messages = [[{"role": "user", "content": [
                {"type": "video", "video": v},
                {"type": "text", "text": prompt}]}] for v in batch]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
                do_sample_frames=False).to(model.device)
            # XLA: 动态 KV cache 每个解码位置都触发重编译(几十张图);
            # static cache 只编 prefill + decode 两张图(真机确认项)
            gen = dict(max_new_tokens=max_new_tokens, do_sample=False)
            try:
                out = model.generate(**inputs,
                                     cache_implementation="static", **gen)
            except Exception as e:
                print(f"[WARN] static cache 失败({str(e)[:80]}),"
                      f"回退动态 cache(XLA 下会很慢)")
                out = model.generate(**inputs, **gen)
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
        batch.append(load_frames_for_record(rec, cfg))   # (T,H,W,3) uint8
        metas.append(vid)
        if len(batch) >= batch_size:
            flush()
    flush()
    fout.close()
    return out_path
