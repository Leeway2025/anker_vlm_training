#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""终版池构建: 批次口音过滤(半池冠军一致率) + test边际对齐 + 尾类全收。
产出 /data/pool_700k_final_labels.jsonl 与 /data/pool_500k_final_labels.jsonl(嵌套)。"""
import json, random, re
from collections import Counter, defaultdict

POOL = '/home/nas-tpu-poc/data/zx_vlm_dataset/anker_video_clips/euno_train_v3.0.24_des25_deduped_fixed_exclude_gen_videos_train_format_v23_frames.json'
TAIL = set("qrujonst"); NONZH = re.compile(r"[一-鿿]")

# ── 批次口音地图(半池自然覆盖, 冠军 SubKS 一致率)──
gt = {}
for l in open('/data/pool_half_b.jsonl'):
    j = json.loads(l); gt[j['video_id']] = j['labels']['sub_keyscene']
bat = defaultdict(lambda: [0, 0])
for l in open('/data/gate2_half_b_scores.jsonl'):
    j = json.loads(l)
    m = re.match(r"\s*([A-E])\s*\|\s*([a-u])\s*\|", j.get('output') or '')
    if not m or j['video_id'] not in gt:
        continue
    b = j['video_id'].split('/')[0]
    bat[b][0] += (m.group(2) == gt[j['video_id']]); bat[b][1] += 1
rates = {b: a / c for b, (a, c) in bat.items() if c >= 300}
med = sorted(rates.values())[len(rates) // 2]
BAD = {b for b, r in rates.items() if r < max(0.60, med - 0.08)}
print(f"[口音] 批次 {len(rates)} 中位 {med:.2f} | 剔除 {len(BAD)}: "
      f"{sorted((b, round(rates[b],2)) for b in BAD)[:10]}")

recs = []
for r in json.load(open(POOL)):
    g = next((c['value'] for c in r['conversations'] if c['from'] == 'gpt'), '')
    m = re.match(r'\s*([A-E])\|([a-u])\|(.*)', g, re.S)
    if not m or not m.group(3).strip() or NONZH.search(m.group(3)):
        continue
    if r['video'].split('/')[0] in BAD:
        continue
    recs.append((r['video'], m.group(1), m.group(2)))
print(f"[池] 口音过滤后 {len(recs)}")

t_rt = Counter()
for l in open('/data/labels_test.jsonl'):
    t_rt[json.loads(l)['labels']['role_type']] += 1
tn = sum(t_rt.values())
random.Random(0).shuffle(recs)
tail = [r for r in recs if r[2] in TAIL]
rest = [r for r in recs if r[2] not in TAIL]
idx = json.load(open('/home/nas-tpu-poc/data/zx_vlm_dataset/anker_video_clips_wds_full/index.json'))
WDS = '/home/nas-tpu-poc/data/zx_vlm_dataset/anker_video_clips_wds_full'
pool_lbl = {}
for l in open('/data/pool_full_labels.jsonl'):
    j = json.loads(l); pool_lbl[j['video_id']] = j['labels']
# pool_full_labels 只有前半…… 补: 直接从 recs 重建行
def row(vid, rt, sk, desc):
    return json.dumps({"video_id": vid, "video_uri": vid,
                       "labels": {"role_type": rt, "sub_keyscene": sk, "description": desc},
                       "meta": {"camera_id": "unknown", "storage": "wds",
                                "wds_dir": WDS, "shard": idx.get(vid, 0)}},
                      ensure_ascii=False)
desc_map = {}
for r in json.load(open(POOL)):
    g = next((c['value'] for c in r['conversations'] if c['from'] == 'gpt'), '')
    m = re.match(r'\s*([A-E])\|([a-u])\|(.*)', g, re.S)
    if m:
        desc_map[r['video']] = m.group(3).strip()

for SIZE in (700000, 500000):
    sel = list(tail)
    got = Counter(rt for _, rt, _ in tail)
    quota = {k: int(SIZE * v / tn) for k, v in t_rt.items()}
    for v, rt, sk in rest:
        if len(sel) >= SIZE: break
        if got[rt] >= quota.get(rt, 0): continue
        got[rt] += 1; sel.append((v, rt, sk))
    seen = {v for v, _, _ in sel}
    for v, rt, sk in rest:
        if len(sel) >= SIZE: break
        if v not in seen: sel.append((v, rt, sk)); seen.add(v)
    out = f'/data/pool_{SIZE//1000}k_final_labels.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for v, rt, sk in sel[:SIZE]:
            f.write(row(v, rt, sk, desc_map.get(v, '')) + '\n')
    print(f"[出池] {out} {min(len(sel),SIZE)} 条 RT配比 {dict(got)}")
