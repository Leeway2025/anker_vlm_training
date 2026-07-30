#!/bin/bash
# CoT 链 ↔ GT 标签全量对账: 解析链内显式字母(RoleType X / Sub-keyscene y)与 GT 硬碰。
# 只读,可与训练并行。用法: bash scripts/check_cot_labels.sh
set -e
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json, re, random, collections

ASSET = '/data/assets_rat/asset_C_reasoning.jsonl'
gt = {}
for l in open('/data/labels_dedup.jsonl', encoding='utf-8'):
    d = json.loads(l); lb = d.get('labels') or d
    gt[d['video_id']] = (lb['role_type'], lb['sub_keyscene'],
                         str(lb.get('description', ''))[:70])

RT_RE = re.compile(r'Role\s*Type\s*[:\-]?\s*\(?([A-Ea-e])\b', re.I)
SK_RE = re.compile(r'Sub[\s\-_]?keyscene\s*[:\-]?\s*\(?([A-Ua-u])\b', re.I)

n = rt_hit = sk_hit = rt_ok = sk_ok = 0
rt_conf = collections.Counter(); sk_conf = collections.Counter()
mism = []
for l in open(ASSET, encoding='utf-8'):
    d = json.loads(l)
    v, c = d['video_id'], d['reasoning_chain']
    if v not in gt: continue
    n += 1
    g_rt, g_sk, desc = gt[v]
    m = RT_RE.search(c)
    if m:
        rt_hit += 1
        p = m.group(1).upper()
        if p == g_rt: rt_ok += 1
        else:
            rt_conf[f'{g_rt}->{p}'] += 1
            mism.append(('RT', v, g_rt, p, desc, c))
    m = SK_RE.search(c)
    if m:
        sk_hit += 1
        p = m.group(1).lower()
        if p == g_sk: sk_ok += 1
        else:
            sk_conf[f'{g_sk}->{p}'] += 1
            mism.append(('SK', v, g_sk, p, desc, c))

print(f'对账样本: {n}(资产 ∩ labels_dedup)')
print(f'RT 显式字母提取率 {rt_hit/n:.1%} | 与 GT 一致 {rt_ok}/{rt_hit} '
      f'({rt_ok/max(rt_hit,1):.2%})')
print(f'SK 显式字母提取率 {sk_hit/n:.1%} | 与 GT 一致 {sk_ok}/{sk_hit} '
      f'({sk_ok/max(sk_hit,1):.2%})')
print('RT 错配 Top:', rt_conf.most_common(8))
print('SK 错配 Top:', sk_conf.most_common(8))

random.seed(0); random.shuffle(mism)
print(f'\n== 错配抽样 {min(10,len(mism))} 条(链说的字母 ≠ GT)==')
for kind, v, g, p, desc, c in mism[:10]:
    print(f'\n[{kind}] {v} GT={g} 链说={p} | {desc}')
    print('  ' + c[:280])

bad = (rt_hit and rt_ok/rt_hit < 0.97) or (sk_hit and sk_ok/sk_hit < 0.97)
print('\n[判读] ' + ('一致率低于97%,错配清单需要处理(见上),别急着信这份资产'
                    if bad else
                    '链内字母与 GT 高度一致(≥97%),CoT 标注方向正确,可放心训'))
PY
