#!/bin/bash
# CoT 资产的 RT 体检: ①GT=A 链的身份线索覆盖率(疗效) ②GT=D 链含 A 型证据词
#   的比例(怯判教材,A→D 病根同谋)。只读,秒级。
set -e
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json, re, random

gt = {}
for l in open('/data/labels_dedup.jsonl', encoding='utf-8'):
    d = json.loads(l); lb = d.get('labels') or d
    gt[d['video_id']] = (lb['role_type'], lb['sub_keyscene'])

A_CUE = re.compile(r'\bresident|home\s?owner|family member|confident(?:ly)?\b|'
                   r'\bfamiliar(?:ity)?\b|his own|her own|returns? home|unlock',
                   re.I)
n = {'A': 0, 'D': 0}; hit = {'A': 0, 'D': 0}
d_bad = []
for l in open('/data/assets_rat/asset_C_reasoning.jsonl', encoding='utf-8'):
    r = json.loads(l)
    v, c = r['video_id'], r['reasoning_chain']
    rt = gt.get(v, ('?',))[0]
    if rt not in n: continue
    n[rt] += 1
    if A_CUE.search(c):
        hit[rt] += 1
        if rt == 'D': d_bad.append((v, gt[v][1], c))

print(f"GT=A 链: {n['A']} 条,含身份线索词 {hit['A']} ({hit['A']/max(n['A'],1):.1%})"
      f"  ← 越高越好(教'认得出家人')")
print(f"GT=D 链: {n['D']} 条,含 A 型证据词 {hit['D']} ({hit['D']/max(n['D'],1):.1%})"
      f"  ← 怯判教材(有A证据仍结论D)")
random.seed(0); random.shuffle(d_bad)
print('\n== GT=D 却满嘴 A 证据的链,抽 8 条 ==')
for v, sk, c in d_bad[:8]:
    print(f'\n[D|{sk}] {v}')
    print('  ' + c[:260])
PY
