#!/bin/bash
# S5 开训前体检: CoT 资产格式/覆盖/泄漏/质量。硬伤退出码非0(挡住夜链)。
# 用法: bash scripts/check_cot_asset.sh && nohup bash scripts/night_s5.sh > night_s5.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json, re, sys, random, collections

ASSET = '/data/assets_rat/asset_C_reasoning.jsonl'
rows, bad_json, bad_field = {}, 0, 0
dups = 0
for l in open(ASSET, encoding='utf-8'):
    try:
        d = json.loads(l)
    except Exception:
        bad_json += 1; continue
    v, c = d.get('video_id'), d.get('reasoning_chain')
    if not v or not isinstance(c, str) or not c.strip():
        bad_field += 1; continue
    if v in rows: dups += 1
    rows[v] = c.strip()
n = len(rows)
print(f'资产行: 有效 {n} | 坏JSON {bad_json} | 缺字段/空链 {bad_field} | 重复id {dups}')

gt = {}
for l in open('/data/labels_dedup.jsonl', encoding='utf-8'):
    d = json.loads(l); lb = d.get('labels') or d
    gt[d['video_id']] = (lb['role_type'], lb['sub_keyscene'], str(lb.get('description',''))[:60])
val = {x.strip() for x in open('/data/val_ids_v2.txt')}
test = {json.loads(l)['video_id'] for l in open('/data/labels_test.jsonl', encoding='utf-8')}
train_ids = set(gt) - val

cov = len(train_ids & set(rows))
leak_test = len(test & set(rows))
orphan = len(set(rows) - set(gt))
print(f'覆盖 train: {cov}/{len(train_ids)} ({cov/len(train_ids):.1%}) | '
      f'孤儿id(不在labels_dedup): {orphan}')
print(f'测试集重叠: {leak_test}  ← 必须为 0')

cjk = sum(1 for c in rows.values() if re.search(r'[一-鿿]', c))
wl = sorted(len(c.split()) for c in rows.values())
q = lambda p: wl[int(p*len(wl))]
short = sum(1 for w in wl if w < 10); long_ = sum(1 for w in wl if w > 120)
print(f'中文混入: {cjk} 条 | 词数 p5/p50/p95 = {q(.05)}/{q(.5)}/{q(.95)} | '
      f'<10词 {short} 条 | >120词 {long_} 条')

BANNED = re.compile(r'ground.?truth|the label|annotat|provided answer|correct answer',
                    re.I)
spill = {v for v, c in rows.items() if BANNED.search(c)}
print(f'泄漏话术(label/ground truth/annotation...): {len(spill)} 条')

# ==== 清洗: 剔除测试集重叠 + 泄漏话术 → asset_C_clean.jsonl(夜链只吃它) ====
CLEAN = '/data/assets_rat/asset_C_clean.jsonl'
clean = {v: c for v, c in rows.items() if v not in test and v not in spill}
with open(CLEAN, 'w', encoding='utf-8') as f:
    for v, c in clean.items():
        f.write(json.dumps({"video_id": v, "reasoning_chain": c},
                           ensure_ascii=False) + '\n')
cov_c = len(train_ids & set(clean))
print(f'清洗: 剔测试重叠 {leak_test} + 泄漏话术 {len(spill)} → 净 {len(clean)} 条'
      f' | 洗后覆盖 train {cov_c}/{len(train_ids)} ({cov_c/len(train_ids):.1%})'
      f' -> {CLEAN}')

random.seed(0)
print('\n== 抽样 5 条(肉眼扫: 链的结论应与 GT 字母同向,不许出现"标签说") ==')
for v in random.sample(sorted(train_ids & set(clean)), 5):
    r, s, desc = gt[v]
    print(f'\n[{v}] GT={r}|{s}| {desc}')
    print('  ' + clean[v][:300])

hard = []
if bad_json + bad_field > 0.05 * max(n, 1): hard.append('坏行超5%')
if leak_test > 0.10 * max(n, 1): hard.append(f'测试集泄漏超10%({leak_test}),生成批次有大错')
if cov_c < 0.5 * len(train_ids): hard.append(f'洗后覆盖率过低 {cov_c/len(train_ids):.1%}')
if cjk > 0.01 * n: hard.append('中文混入超1%')
if hard:
    print(f'\n[硬伤,拒绝开训] {hard}'); sys.exit(1)
print(f'\n[体检通过] 洗后覆盖 {cov_c/len(train_ids):.1%};夜链将使用 {CLEAN}')
if leak_test:
    print(f'[遗留议题] labels_dedup ∩ labels_test = 训练池含测试视频(资产泄漏{leak_test}条'
          f'即由此而来)——白天必须专题处理,今晚仅隔离资产不动训练池')
PY
