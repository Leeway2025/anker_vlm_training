#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按三层口径把 GT描述judge 的错标应用到训练集(修正为主、删除为辅)。
输入: judge(gt_desc_judge_train) + blind(euno_train balanced) + model(kto_prep/train_preds)。
口径(用户 08-03 拍板):
  高置信 = judge判mislabel 且 correct_rt 被 blind 或 model 至少一个同向证实 → 【修正】label.role_type = correct_rt
  低置信 = judge-only(无任何证人同向) → 【删除】该样本(吃不准的不猜标签, 去噪即可)
  C: judge 本就只升不降、correct_rt∈{A,B,C,E}, 不会降C/入D。
产出(原文件不动, 只写新文件, 全程可回溯):
  --clean   清洗后训练集(改的改、删的删)
  gt_train_corrections.jsonl / gt_train_deletes.jsonl  两份明细
用法: python3 scripts/apply_gt_fixes.py --judge /data/gt_desc_judge_train.jsonl \
        --labels /data/labels_dedup.jsonl --clean /data/labels_dedup_clean.jsonl
"""
import argparse, json, os
from collections import Counter


def load_rt(path, kind):
    o = {}
    if not os.path.exists(path):
        return o
    for l in open(path, encoding="utf-8"):
        try: d = json.loads(l)
        except: continue
        v = d.get("video_id")
        if not v: continue
        if kind == "blind":
            g = (d.get("gemini_output") or d).get("predictions") or {}
            o[v] = g.get("role_type")
        else:  # model
            seg = (d.get("output") or "").split("|")
            o[v] = seg[0].strip() if seg else None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="/data/gt_desc_judge_train.jsonl")
    ap.add_argument("--labels", default="/data/labels_dedup.jsonl")
    ap.add_argument("--clean", default="/data/labels_dedup_clean.jsonl")
    ap.add_argument("--blind", default="/home/nas-tpu-poc/data/zx_vlm_dataset/anker_video_clips/euno_train_v3.0.18_balanced_100k_labels.jsonl")
    ap.add_argument("--model", default="outputs/kto_prep/train_preds.jsonl")
    ap.add_argument("--diag", default="outputs/diag")
    a = ap.parse_args()

    blind = load_rt(a.blind, "blind"); model = load_rt(a.model, "model")
    print(f"[witness] blind={len(blind)} model={len(model)}")

    correct, delete = {}, {}           # vid -> (correct_rt / reason)
    mis = 0
    for l in open(a.judge, encoding="utf-8"):
        d = json.loads(l)
        if d["verdict"] != "mislabel":
            continue
        mis += 1
        v = d["video_id"]; cr = d["correct_rt"]; gr = d["gt_rt"]
        b = blind.get(v); m = model.get(v)
        corrob = [w for w, x in (("blind", b), ("model", m)) if x == cr]
        rec = {"video_id": v, "gt_rt": gr, "correct_rt": cr, "cue": d.get("cue", ""),
               "blind": b, "model": m, "corrob": corrob, "desc": d.get("desc", "")}
        if corrob:                      # 高置信 → 修正
            correct[v] = rec
        else:                           # judge-only → 删除
            delete[v] = rec

    os.makedirs(a.diag, exist_ok=True)
    with open(f"{a.diag}/gt_train_corrections.jsonl", "w", encoding="utf-8") as f:
        for r in correct.values(): f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{a.diag}/gt_train_deletes.jsonl", "w", encoding="utf-8") as f:
        for r in delete.values(): f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_in = n_corr = n_del = 0
    with open(a.labels, encoding="utf-8") as fin, open(a.clean, "w", encoding="utf-8") as fout:
        for l in fin:
            d = json.loads(l); v = d["video_id"]; n_in += 1
            if v in delete:
                n_del += 1; continue
            if v in correct:
                d["labels"]["role_type"] = correct[v]["correct_rt"]
                d.setdefault("meta", {})["rt_fixed_by"] = "gt_desc_judge_3tier"
                n_corr += 1
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    cd = Counter(f"{r['gt_rt']}→{r['correct_rt']}" for r in correct.values())
    dd = Counter(f"{r['gt_rt']}→{r['correct_rt']}" for r in delete.values())
    print(f"[judge] mislabel {mis}  = 修正 {len(correct)} + 删除 {len(delete)}")
    print(f"[修正方向] {dict(cd.most_common())}")
    print(f"[删除方向] {dict(dd.most_common())}")
    print(f"[clean] 原 {n_in} → 修正 {n_corr} 删除 {n_del} → 保留 {n_in-n_del} 条 -> {a.clean}")
    print(f"        明细: {a.diag}/gt_train_corrections.jsonl / gt_train_deletes.jsonl")


if __name__ == "__main__":
    main()
