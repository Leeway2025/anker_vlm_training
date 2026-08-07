#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1M 池 json → 我方 labels jsonl(闸2 打分/训练通用)。--ids 只转清单内。"""
import argparse, json, re
ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--ids", default="", help="可选: 只转此清单内 video_id")
ap.add_argument("--wds-dir", default="/home/nas-tpu-poc/data/zx_vlm_dataset/anker_video_clips_wds_full")
ap.add_argument("--out", required=True)
a = ap.parse_args()
scope = {x.strip() for x in open(a.ids)} if a.ids else None
import os
idx = {}
ip = os.path.join(a.wds_dir, "index.json")
if os.path.exists(ip):
    idx = json.load(open(ip))
    print(f"[index] {len(idx)} 条")
raw = json.load(open(a.pool, encoding="utf-8"))
n = 0
with open(a.out, "w", encoding="utf-8") as f:
    for r in raw:
        vid = r["video"]
        if scope is not None and vid not in scope:
            continue
        g = next((c["value"] for c in r["conversations"] if c["from"] == "gpt"), "")
        m = re.match(r"\s*([A-E])\|([a-u])\|(.*)", g, re.S)
        if not m:
            continue
        f.write(json.dumps({
            "video_id": vid, "video_uri": vid,
            "labels": {"role_type": m.group(1), "sub_keyscene": m.group(2),
                       "description": m.group(3).strip()},
            "meta": {"camera_id": "unknown", "storage": "wds",
                     "wds_dir": a.wds_dir,
                     "shard": idx.get(vid, 0)}}, ensure_ascii=False) + "\n")
        n += 1
print(f"[done] {n} -> {a.out}")
