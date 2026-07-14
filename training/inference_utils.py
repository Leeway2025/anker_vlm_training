"""批量推理工具(hard_mining / build_kto_data / 评测共用)。

帧来源存储自适应(euno_wds.load_frames_for_record):
  meta.storage=='wds'(客户数据)→ 分片直读;否则视频文件解码。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.euno_wds import load_frames_for_record  # noqa: E402

try:
    import torch_xla.core.xla_model as xm
    _XLA = True
except ImportError:
    _XLA = False


class _XlaStepMarker:
    """LogitsProcessor: 每个解码步前向后立刻切图(勿删)。

    transformers 的 generate 循环每步都要读停止条件的张量值
    (stopping_criteria / _has_unfinished_sequences),而解码步之间
    没有任何图边界 → 惰性图逐 token 增长,每 token 强制编译一张
    更大的图(v6e 实测: 4 样本 50 分钟 0 输出,22 张图越编越大)。
    LogitsProcessor 恰好在每步前向之后、停止判断取值之前被调用:
    在这里 mark_step,图上限 = 单个解码步;配合 static cache,
    全程仅编译 prefill + decode 两张图,之后逐 token 复用。"""

    def __call__(self, input_ids, scores):
        if _XLA:
            xm.mark_step()
        return scores


def shard_records(records, spec):
    """'i/n' → 第 i 片(0 起)。大规模推理按分片多进程/多机并行,
    各片独立断点续跑(out 文件不同名)。"""
    if not spec:
        return records
    i, n = (int(x) for x in spec.split("/"))
    return records[i::n]


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
            from transformers import LogitsProcessorList
            gen = dict(max_new_tokens=max_new_tokens, do_sample=False,
                       logits_processor=LogitsProcessorList([_XlaStepMarker()]))
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
