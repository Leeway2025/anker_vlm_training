#!/bin/bash
# 1M P2b: 盲判×GT 分层错标率地图(纯CPU)
# 用法: bash scripts/pool_blind_report.sh /path/labels_1m.jsonl [盲判文件]
set -e
cd "$(dirname "$0")/.."
LB="${1:?用法: pool_blind_report.sh <labels_1m.jsonl> [blind.jsonl]}"
BL="${2:-/data/pool_blind.jsonl}"
LB="$LB" BL="$BL" python3 - <<'PY'
import json, os, collections
gt = {}
for l in open(os.environ['LB'], encoding='utf-8'):
    d = json.loads(l); lb = d.get('labels') or d
    gt[d['video_id']] = (lb.get('role_type'), lb.get('sub_keyscene'))
n = collections.Counter(); ag_sk = collections.Counter(); ag_rt = collections.Counter()
conf = collections.defaultdict(collections.Counter)
for l in open(os.environ['BL'], encoding='utf-8'):
    d = json.loads(l)
    g = (d.get('gemini_output') or {}).get('predictions') or {}
    v = d['video_id']
    if v not in gt or not g.get('sub_keyscene'):
        continue
    grt, gsk = gt[v]
    n[gsk] += 1
    ag_sk[gsk] += (g['sub_keyscene'] == gsk)
    ag_rt[gsk] += (g.get('role_type') == grt)
    if g['sub_keyscene'] != gsk:
        conf[gsk][g['sub_keyscene']] += 1
tot = sum(n.values())
print(f'盲判样本 {tot} 条 | 总体 SK 一致率 {sum(ag_sk.values())/max(tot,1):.1%} '
      f'| RT 一致率 {sum(ag_rt.values())/max(tot,1):.1%}')
print(f"{'类':<4}{'n':>6}{'SK一致':>8}{'RT一致':>8}  主要分歧")
for sk in sorted(n, key=lambda k: ag_sk[k]/max(n[k],1)):
    top = ' '.join(f'{k}={v}' for k, v in conf[sk].most_common(3))
    flag = ' ←重点抽查' if ag_sk[sk]/max(n[sk],1) < 0.55 else ''
    print(f'{sk:<4}{n[sk]:>6}{ag_sk[sk]/max(n[sk],1):>8.1%}'
          f'{ag_rt[sk]/max(n[sk],1):>8.1%}  {top}{flag}')
print('\n[判读] 参照100k经验: 一致率~37%为盲判正常水位(盲判偏简单样本),'
      '显著低于同类100k水位的类=标注质量恶化信号;不做整类清洗,只人工抽查')
PY
