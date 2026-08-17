#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""700k 稀有类加权(增强)权重集(用户 0811 "增强先100k再700k" + "尽量排满")。
在已验证的 702k 全池 labels_train_plus_testval_v2.jsonl 上:
  - 直接用整份文件作 --labels(train 702k + val 2000,val 命中 test_val_ids_v2 不加权);
  - 按 702k 池自身的 SubKS 类频重算 hard-mining 权重(与 100k 同口径:
    w = clip(sqrt(median/count),1,CAP);安全 qrunj 地板 3.0;尾类 o/s/t 地板 2.0)。
只写 w>1 的 video_id → /data/sw_rare_700k.json;train_sft 仅对 train 行(append val 前)生效。
100k 上该方案已验:SubKS +0.37 / RT+SubKS +0.53 / 安全召回 +1.67。"""
import json, math
from collections import Counter

SRC = '/data/labels_train_plus_testval_v2.jsonl'
VALIDS = '/data/test_val_ids_v2.txt'
SW = '/data/sw_rare_700k.json'
CAP = 4.0
SAFETY = set("qrunj"); SAFETY_FLOOR = 3.0
TAIL = set("qrujonst"); TAIL_FLOOR = 2.0

val_ids = set(x.strip() for x in open(VALIDS) if x.strip())
train_rows = []
for l in open(SRC, encoding='utf-8'):
    try:
        d = json.loads(l)
    except Exception:
        continue
    if d.get('video_id') in val_ids:
        continue
    train_rows.append(d)
print(f"[data] train={len(train_rows)} (val {len(val_ids)} 不加权)", flush=True)

cnt = Counter(d['labels']['sub_keyscene'] for d in train_rows)
med = sorted(cnt.values())[len(cnt) // 2]
print(f"[subks] classes={len(cnt)} median={med}", flush=True)

def wfor(cls):
    w = min(CAP, math.sqrt(med / cnt[cls]))
    if cls in SAFETY:
        w = max(w, SAFETY_FLOOR)
    elif cls in TAIL:
        w = max(w, TAIL_FLOOR)
    return max(1.0, w)

wcls = {c: round(wfor(c), 3) for c in cnt}
sw = {}
for d in train_rows:
    c = d['labels']['sub_keyscene']
    if wcls[c] > 1.001:
        sw[d['video_id']] = wcls[c]
json.dump(sw, open(SW, 'w'))

grown = sum(wcls[d['labels']['sub_keyscene']] for d in train_rows)
print(f"[weight] 加权样本={len(sw)}  复制后≈{grown:,.0f} (x{grown/len(train_rows):.2f})", flush=True)
print("[weight] 每类 (count, weight):")
for c in sorted(cnt, key=lambda x: cnt[x]):
    tag = '★安全' if c in SAFETY else ('·尾' if c in TAIL else '')
    print(f"   {c}: n={cnt[c]:>7}  w={wcls[c]:.3f} {tag}")
print(f"[done] {SW}", flush=True)
