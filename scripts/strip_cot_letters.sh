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
RT = r'(?i:Role\s*Type)'; SK = r'(?i:Sub[\s\-_]?keyscene)'
# 剥除模式(按序): 答案字母指纹 RoleType[A-E]/Sub-keyscene[a-u] 的各种写法。
# 短语用作用域标志 (?i:) 忽略大小写("role type"/"RoleType" 通吃),字母限
# 大写[A-E]/小写[a-u] 不加 re.I —— 否则会误吃冠词 "a"(role type as *a* resident)。
PATS = [
    # ⓪ 整括号内含答案字母 → 整括号删(吃 "(Staff, RoleType B)" 这类带前缀括号)
    re.compile(rf'\([^()]*(?:{RT}[\s:,\-=]*\(?\s*[A-E]|{SK}[\s:,\-=]*\(?\s*[a-u])[^()]*\)'),
    # ① RT 短语(+连接词)(+括号)+ 大写字母
    re.compile(rf'{RT}[\s:,\-=]*(?:is|as|of|type)?\s*\(?\s*[A-E]\s*\)?(?=\W|$)'),
    # ② SK 短语(+括号/紧邻/冒号)+ 小写字母(不含 is/of,避免吃冠词)
    re.compile(rf'{SK}[\s:,\-=]*\(?\s*[a-u]\s*\)?(?=\W|$)'),
    # ③ 场景名后独立括号答案码 (a)/(i)/(p)/(E)
    re.compile(r'\(\s*[A-Ea-u]\s*\)'),
    # ④ 裸短语紧跟字母
    re.compile(rf'{RT}\s*[A-E]\b'),
    re.compile(rf'{SK}\s*[a-u]\b'),
]
# 验收器(带字母才算泄漏): 无字母的 "sub-keyscene/role type" 短语提及不计入
LEAK = re.compile(
    rf'{RT}[\s:,\-=]*(?:is|as|of)?\s*\(?\s*[A-E]\b'
    rf'|{SK}[\s:,\-=]*\(?\s*[a-u]\)|{SK}\s+[a-u]\b|\(\s*[A-Ea-u]\s*\)')

def strip(c):
    for p in PATS:
        c = p.sub(' ', c)
    c = re.sub(r'\((?=[A-Za-z][^()]{0,30},)', '', c)   # 孤立左括号 "(Staff," -> "Staff,"
    c = re.sub(r'\(\s*[,;]?\s*\)', '', c)              # 空/半空括号
    c = re.sub(r'\s{2,}', ' ', c).replace(' ,', ',').replace(' .', '.') \
         .replace('( ', '(').strip()
    return c

n = hit = 0
res = []
with open(OUT, 'w', encoding='utf-8') as f:
    for l in open(IN, encoding='utf-8'):
        d = json.loads(l); c = d['reasoning_chain']; n += 1
        c2 = strip(c)
        if c2 != c: hit += 1
        if len(c2.split()) < 10:      # 剥完只剩壳的链直接弃(答案就是它的全部)
            continue
        f.write(json.dumps({"video_id": d['video_id'], "reasoning_chain": c2},
                           ensure_ascii=False) + '\n')
        res.append(c2)
print(f'[strip] {n} 条: 触及 {hit} 条({hit/max(n,1):.1%}),'
      f'剥后过短弃 {n-len(res)} 条,产出 {len(res)} -> {OUT}')
残留 = sum(1 for c in res if LEAK.search(c))
print(f'[验收] 残留答案字母指纹: {残留} 条(必须=0);抽样:')
for c in res[:3]:
    print('  ' + c[:200])
if 残留:
    raise SystemExit(f'[FATAL] 去字母化残留 {残留} 条带字母泄漏,拒绝产出')
PY
