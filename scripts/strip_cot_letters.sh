#!/bin/bash
# CoT 资产去字母化(CPU 秒级): 剥掉链内答案指纹(RoleType X / Sub-keyscene y /
#   括号字母结论),保留证据描述 —— S5 泄题死因的手术刀。任何 CoT 重训前必过。
# 用法: bash scripts/strip_cot_letters.sh [输入资产] [输出资产]
#   默认: /data/assets_rat/asset_C_reasoning.jsonl -> asset_C_nolttr.jsonl
set -e
cd "$(dirname "$0")/.."
IN="${1:-/data/assets_rat/asset_C_reasoning.jsonl}"
OUT="${2:-/data/assets_rat/asset_C_nolttr.jsonl}"
IN="$IN" OUT="$OUT" python3 - <<'PY'
import json, os, re

IN, OUT = os.environ['IN'], os.environ['OUT']
# 剥除模式(大小写不敏感): 括号版/裸版的 RoleType X 与 Sub-keyscene y,
# 以及 "(RoleType D, Sub-keyscene c)" 联合括号
PATS = [
    re.compile(r'\s*\(\s*Role\s*Type\s*[A-E]\s*(?:,\s*Sub[\s\-_]?keyscene\s*[a-uA-U]\s*)?\)', re.I),
    re.compile(r'\s*\(\s*Sub[\s\-_]?keyscene\s*[a-uA-U]\s*(?:,\s*Role\s*Type\s*[A-E]\s*)?\)', re.I),
    re.compile(r'\bRole\s*Type\s*[A-E]\b', re.I),
    re.compile(r'\bSub[\s\-_]?keyscene\s*[a-uA-U]\b', re.I),
]
n = hit = 0
res = []
with open(OUT, 'w', encoding='utf-8') as f:
    for l in open(IN, encoding='utf-8'):
        d = json.loads(l); c = d['reasoning_chain']; n += 1
        c2 = c
        for p in PATS:
            c2 = p.sub('', c2)
        c2 = re.sub(r'\s{2,}', ' ', c2).replace(' ,', ',').replace(' .', '.').strip()
        if c2 != c: hit += 1
        if len(c2.split()) < 10:      # 剥完只剩壳的链直接弃(答案就是它的全部)
            continue
        f.write(json.dumps({"video_id": d['video_id'], "reasoning_chain": c2},
                           ensure_ascii=False) + '\n')
        res.append(c2)
print(f'[strip] {n} 条: 剥除字母 {hit} 条({hit/max(n,1):.1%}),'
      f'剥后过短弃 {n-len(res)} 条,产出 {len(res)} -> {OUT}')
残留 = sum(1 for c in res if re.search(r'Role\s*Type|Sub[\s\-_]?keyscene', c, re.I))
print(f'[验收] 残留答案指纹: {残留} 条(必须=0);抽样:')
for c in res[:3]:
    print('  ' + c[:200])
PY
