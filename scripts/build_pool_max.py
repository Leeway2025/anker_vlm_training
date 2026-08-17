#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最大有用池: 与 build_pool_final 同质量闸(口音过滤 + 结构闸),但不加 700k 上限、
不做 RT 配额下采样 —— 收下全部通过质量闸的片段(~923k)。再把当前 v2 训练文件里的
2000 条 val 行原样附加,复用 test_val_ids_v2.txt 做 val,保证与 anneal_b 口径一致。
产出 /data/labels_max_natural.jsonl。"""
import json, re
from collections import Counter, defaultdict

POOL = '/home/nas-tpu-poc/data/zx_vlm_dataset/anker_video_clips/euno_train_v3.0.24_des25_deduped_fixed_exclude_gen_videos_train_format_v23_frames.json'
CUR = '/data/labels_train_plus_testval_v2.jsonl'
VALIDS = '/data/test_val_ids_v2.txt'
OUT = '/data/labels_max_natural.jsonl'
WDS = '/home/nas-tpu-poc/data/zx_vlm_dataset/anker_video_clips_wds_full'
NONZH = re.compile(r"[一-鿿]")

# ── 口音闸(与 build_pool_final 完全一致)──
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
print(f"[accent] batches={len(rates)} median={med:.3f} dropped={len(BAD)}", flush=True)

idx = json.load(open(f'{WDS}/index.json'))

def row(vid, rt, sk, desc):
    return json.dumps({"video_id": vid, "video_uri": vid,
                       "labels": {"role_type": rt, "sub_keyscene": sk, "description": desc},
                       "meta": {"camera_id": "unknown", "storage": "wds",
                                "wds_dir": WDS, "shard": idx.get(vid, 0)}},
                      ensure_ascii=False)

n = 0
rt_c = Counter()
seen = set()
with open(OUT, 'w', encoding='utf-8') as f:
    for r in json.load(open(POOL)):
        g = next((c['value'] for c in r['conversations'] if c['from'] == 'gpt'), '')
        m = re.match(r'\s*([A-E])\|([a-u])\|(.*)', g, re.S)
        if not m or not m.group(3).strip() or NONZH.search(m.group(3)):
            continue
        vid = r['video']
        if vid.split('/')[0] in BAD:
            continue
        f.write(row(vid, m.group(1), m.group(2), m.group(3).strip()) + '\n')
        seen.add(vid); rt_c[m.group(1)] += 1; n += 1
    print(f"[pool] quality-gated train rows = {n}  RT={dict(sorted(rt_c.items()))}", flush=True)

    # 附加 2000 条 val 行(从当前 v2 文件抽 test_val_ids_v2 命中的行,口径不变)
    val_ids = set(x.strip() for x in open(VALIDS) if x.strip())
    nv = 0
    for l in open(CUR):
        try:
            d = json.loads(l)
        except Exception:
            continue
        vid = d.get('video_id')
        if vid in val_ids and vid not in seen:
            f.write(l if l.endswith('\n') else l + '\n')
            seen.add(vid); nv += 1
    print(f"[val] appended val rows = {nv} (of {len(val_ids)} val ids)", flush=True)

print(f"[done] {OUT} total = {n + nv}", flush=True)
