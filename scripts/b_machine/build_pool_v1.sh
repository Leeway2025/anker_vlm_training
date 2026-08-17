#!/bin/bash
# 1M P3: 策展出池: 去重+坏行剔除+自然分布保留(不配平/不剔测试重叠)
#   + 1M 专属 val 卷(test-mix 1536) + 血统档案
# 用法: bash scripts/build_pool_v1.sh /path/labels_1m_raw.jsonl
set -e
cd "$(dirname "$0")/.."
RAW="${1:?用法: build_pool_v1.sh <labels_1m_raw.jsonl>}"
OUT=/data/labels_1m_v1.jsonl

RAW="$RAW" OUT="$OUT" python3 - <<'PY'
import json, os, re
RT, SK = 'ABCDE', 'abcdefghijklmnopqrstu'
CJK = re.compile(r'[一-鿿]')
seen = set()
stats = dict(rows=0, dup=0, badlab=0, empty=0, cjk=0, keep=0)
with open(os.environ['OUT'], 'w', encoding='utf-8') as f:
    for l in open(os.environ['RAW'], encoding='utf-8'):
        try:
            d = json.loads(l)
        except Exception:
            stats['badlab'] += 1; continue
        lb = d.get('labels') or d
        v = d.get('video_id'); stats['rows'] += 1
        if not v or lb.get('role_type') not in RT or lb.get('sub_keyscene') not in SK:
            stats['badlab'] += 1; continue
        if v in seen:
            stats['dup'] += 1; continue
        desc = str(lb.get('description', '')).strip()
        if not desc:
            stats['empty'] += 1; continue
        if CJK.search(desc):
            stats['cjk'] += 1; continue     # 中文desc先剔出池,单独走翻译轮(见PLAN_1M P3.5)
        seen.add(v)
        f.write(l if l.endswith('\n') else l + '\n')
        stats['keep'] += 1
print(f"[v1] {stats} -> {os.environ['OUT']}")
print(f"[血统] 源={os.environ['RAW']} 规则=去重+坏行/空desc/中文desc剔除,无配平,无重叠剔除")
PY

# 1M 专属 val 卷(缺口驱动 test-mix 装箱,幂等)
if [ ! -s /data/val_ids_1m.txt ]; then
  python3 jax_impl/export_val_split.py --labels $OUT \
      --val-n 1536 --seed 0 --match-mix /data/labels_test.jsonl --fill-loose \
      --out /data/labels_val_1m.jsonl --ids-out /data/val_ids_1m.txt
fi
echo "[OK] 池=$OUT val卷=/data/val_ids_1m.txt;血统信息请一并存入 logs/README"
