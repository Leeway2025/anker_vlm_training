#!/bin/bash
# RT 版失分归因: 对交付口径的 RoleType 错误分桶(考卷/模型/真难)
# 用法: bash scripts/rt_attribute.sh   (纯 CPU,10 秒)
set -e
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json, collections
lab = {}
for l in open('/data/labels_test.jsonl'):
    d = json.loads(l); v = d.get('labels') or d
    lab[d['video_id']] = v['role_type']
md = {}
for l in open('outputs/optin/preds_surg.jsonl'):
    d = json.loads(l); md[d['video_id']] = d['output'].split('|')[0].strip()
bl = {}
for l in open('/data/test_blind.jsonl'):
    d = json.loads(l); p = (d.get('gemini_output') or d).get('predictions') or {}
    if p.get('role_type'): bl[d['video_id']] = p['role_type']
buck = collections.Counter(); pair = collections.Counter()
byc = collections.defaultdict(collections.Counter)
tot = err = 0
for v, g in lab.items():
    m = md.get(v)
    if m is None: continue
    tot += 1
    if m == g: continue
    err += 1; pair[(g, m)] += 1
    b = bl.get(v)
    k = ('G_证人缺席' if b is None else
         'A_考卷重嫌' if b == m else
         'E_模型缺陷' if b == g else 'F_真难')
    buck[k] += 1; byc[k][g] += 1
print(f'RT 错误 {err}/{tot}(acc={1-err/tot:.2%})\n')
for k in ['A_考卷重嫌', 'E_模型缺陷', 'F_真难', 'G_证人缺席']:
    n = buck[k]
    top = ' '.join(f'{c}={x}' for c, x in byc[k].most_common(3))
    print(f'{k:<10} {n:>5}({n/max(err,1):>4.0%}) 折分{n/tot*100:5.2f}  GT主类: {top}')
print('\nTop 混淆对(GT→预测):')
for (g, m), n in pair.most_common(8):
    print(f'  {g}→{m}: {n}')
PY
