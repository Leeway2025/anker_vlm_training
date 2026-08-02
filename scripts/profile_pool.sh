#!/bin/bash
# 1M 数据池体检(纯 CPU,只读 labels jsonl,不碰帧): 规模/去重/测试集污染/
#   分布对齐/懒标浓度/desc质量 六项一次出报告。
# 用法: bash scripts/profile_pool.sh /path/to/labels_1m.jsonl [报告输出路径]
set -e
cd "$(dirname "$0")/.."
IN="${1:?用法: profile_pool.sh <labels.jsonl> [report.txt]}"
OUT="${2:-${IN%.jsonl}_profile.txt}"
IN="$IN" python3 - <<'PY' | tee "$OUT"
import json, os, re, collections

IN = os.environ['IN']
RT_SET, SK_SET = 'ABCDE', 'abcdefghijklmnopqrstu'
ID_WORDS = re.compile(r'resident|home\s?owner|courier|delivery|staff|stranger|'
                      r'visitor|intruder|uniform|family|neighbor', re.I)
CJK = re.compile(r'[一-鿿]')

rows, badj, badlab, dup = 0, 0, 0, 0
seen = set()
rtsk = collections.Counter(); rt_c = collections.Counter()
desc_len = []; desc_empty = 0; desc_cjk = 0
d_with_id = 0; d_total = 0; m_total = 0; m_short = 0
cam = collections.Counter()
for l in open(IN, encoding='utf-8'):
    try:
        d = json.loads(l)
    except Exception:
        badj += 1; continue
    lb = d.get('labels') or d
    v = d.get('video_id')
    rt, sk = lb.get('role_type'), lb.get('sub_keyscene')
    if not v or rt not in RT_SET or sk not in SK_SET:
        badlab += 1; continue
    rows += 1
    if v in seen: dup += 1
    seen.add(v)
    rtsk[(rt, sk)] += 1; rt_c[rt] += 1
    desc = str(lb.get('description', '')).strip()
    if not desc: desc_empty += 1
    else:
        desc_len.append(len(desc.split()))
        if CJK.search(desc): desc_cjk += 1
    if rt == 'D':
        d_total += 1
        if ID_WORDS.search(desc): d_with_id += 1
    if sk == 'm':
        m_total += 1
        if len(desc.split()) < 6: m_short += 1
    seg = v.split('/')[-1].split('_')
    cam[seg[1] if len(seg) > 1 else seg[0]] += 1

print(f'== 池体检: {IN}')
print(f'规模: 有效 {rows} | 坏JSON {badj} | 坏标签 {badlab} | 重复id {dup} '
      f'| 独立机位(粗) {len(cam)}')

test = set()
if os.path.exists('/data/labels_test.jsonl'):
    test = {json.loads(l)['video_id'] for l in open('/data/labels_test.jsonl')}
    ov = len(seen & test)
    print(f'★ 测试集污染: {ov} 条 —— v6 构建时硬排除(本次红线,勿重蹈 100k 覆辙)')
old = set()
if os.path.exists('/data/labels_dedup.jsonl'):
    old = {json.loads(l)['video_id'] for l in open('/data/labels_dedup.jsonl')}
    print(f'与现 100k 池重叠: {len(seen & old)} 条(增量净新 = {rows - len(seen & old)})')

print(f'\nRT 分布: ' + ' '.join(f'{k}={100*rt_c[k]/rows:.1f}%' for k in RT_SET))
tt = collections.Counter()
if test:
    for l in open('/data/labels_test.jsonl'):
        lb = json.loads(l); lab = lb.get('labels') or lb
        tt[lab['role_type']] += 1
    nt = sum(tt.values())
    print(f'测试 RT 分布: ' + ' '.join(f'{k}={100*tt[k]/nt:.1f}%' for k in RT_SET)
          + '  ← 差>5个点的类是配平/偏采样嫌疑')
top = rtsk.most_common(8)
print('RT×SK Top8: ' + ' '.join(f'{r}|{s}={n}' for (r, s), n in top))

dl = sorted(desc_len)
q = lambda p: dl[int(p*len(dl))] if dl else 0
print(f'\ndesc: 空 {desc_empty} | 中文 {desc_cjk} | 词数 p5/p50/p95 = '
      f'{q(.05)}/{q(.5)}/{q(.95)}')
print(f'懒标指纹: GT=D 且 desc 含身份词 {d_with_id}/{d_total} '
      f'({100*d_with_id/max(d_total,1):.1f}%,100k池为25.4%) | '
      f'm 类 desc<6词 {m_short}/{m_total} ({100*m_short/max(m_total,1):.1f}%)')
print(f'机位 Top5 集中度: {sum(n for _, n in cam.most_common(5))/rows:.1%}'
      f'(过高=少数机位刷屏,训练易背景过拟合)')
print('\n[处方] 硬排除测试集重叠 → 去重 → 空desc/坏行剔除 → 分布差>5点的类查采样'
      ' → 懒标浓度超30%的类进抽查包;清洗只删铁证,懒标对齐考卷者保留(m/D教训)')
PY
echo "[OK] 报告 -> $OUT"
