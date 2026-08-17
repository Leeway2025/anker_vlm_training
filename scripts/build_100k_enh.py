#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""100k 稀有类加权(增强)试验集(用户 0811 07:20 "增强先来100k上试完再到700k上试")。
从已验证的 702k 池 labels_train_plus_testval_v2.jsonl 里:
  - val 行(test_val_ids_v2 命中)全保留 → 与 anneal_b 同卷子;
  - train 行确定性洗牌后取 10 万条 → /data/labels_100k_v2.jsonl。
再按 SubKS 类稀有度算 hard-mining 权重(w>1 = 物理复制,见 data.py 双语义):
  w = clip(sqrt(median_count / count[class]), 1, CAP);  安全类 qrunj 地板 3.0。
只写 w>1 的 video_id → /data/sw_rare_100k.json。基线臂不带该文件即为对照。"""
import json, random, math
from collections import Counter

SRC = '/data/labels_train_plus_testval_v2.jsonl'
VALIDS = '/data/test_val_ids_v2.txt'
OUT = '/data/labels_100k_v2.jsonl'
SW = '/data/sw_rare_100k.json'
N_TRAIN = 100_000
CAP = 4.0
SAFETY = set("qrunj")      # 安全关键类,地板 3.0
SAFETY_FLOOR = 3.0
TAIL = set("qrujonst")     # SubKS 尾部类(瓶颈),非安全部分地板 2.0
TAIL_FLOOR = 2.0

val_ids = set(x.strip() for x in open(VALIDS) if x.strip())
val_rows, train_rows = [], []
for l in open(SRC, encoding='utf-8'):
    try:
        d = json.loads(l)
    except Exception:
        continue
    (val_rows if d.get('video_id') in val_ids else train_rows).append((d, l))

random.Random(0).shuffle(train_rows)
train_rows = train_rows[:N_TRAIN]
print(f"[data] train={len(train_rows)} val={len(val_rows)}", flush=True)

with open(OUT, 'w', encoding='utf-8') as f:
    for _, l in train_rows:
        f.write(l if l.endswith('\n') else l + '\n')
    for _, l in val_rows:
        f.write(l if l.endswith('\n') else l + '\n')

# ── SubKS 类计数(仅 100k train)──
cnt = Counter(d['labels']['sub_keyscene'] for d, _ in train_rows)
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
for d, _ in train_rows:
    c = d['labels']['sub_keyscene']
    w = wcls[c]
    if w > 1.001:
        sw[d['video_id']] = w
json.dump(sw, open(SW, 'w'))

# 估算复制后 train 规模
grown = sum(wcls[d['labels']['sub_keyscene']] for d, _ in train_rows)
print(f"[weight] 加权样本={len(sw)}  复制后≈{grown:,.0f} (x{grown/len(train_rows):.2f})", flush=True)
print("[weight] 每类 (count, weight):")
for c in sorted(cnt, key=lambda x: cnt[x]):
    tag = '★安全' if c in SAFETY else ''
    print(f"   {c}: n={cnt[c]:>6}  w={wcls[c]:.3f} {tag}")
print(f"[done] {OUT} + {SW}", flush=True)
