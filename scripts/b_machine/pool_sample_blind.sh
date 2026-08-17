#!/bin/bash
# 1M P2: 分层抽样(每 SK 类≤1200,约2万条)→ Gemini 盲判(断点续跑)
# 前置: export GEMINI_API_KEY=AQ.xxx; export WDS_1M=<1M WDS目录>
# 用法: bash scripts/pool_sample_blind.sh /path/labels_1m.jsonl
set -e
cd "$(dirname "$0")/.."
LB="${1:?用法: pool_sample_blind.sh <labels_1m.jsonl>}"
: "${WDS_1M:?请先 export WDS_1M=<1M WDS目录>}"

LB="$LB" python3 - <<'PY'
import json, os, random, collections
random.seed(0)
by_sk = collections.defaultdict(list)
for l in open(os.environ['LB'], encoding='utf-8'):
    d = json.loads(l); lb = d.get('labels') or d
    if lb.get('sub_keyscene'):
        by_sk[lb['sub_keyscene']].append(d['video_id'])
ids = []
for sk, vs in sorted(by_sk.items()):
    random.shuffle(vs)
    ids += vs[:1200]
open('/data/pool_blind_ids.txt', 'w').write('\n'.join(ids))
print(f'[sample] {len(ids)} 条(每类≤1200)-> /data/pool_blind_ids.txt')
print('类量:', {k: min(len(v), 1200) for k, v in sorted(by_sk.items())})
PY

python3 -m annotation.label_euno_wds \
    --wds-dir "$WDS_1M" --only-ids /data/pool_blind_ids.txt \
    --model "${RAT_MODEL:-gemini-3.1-pro-preview}" --workers 50 \
    --out /data/pool_blind.jsonl
echo "[OK] 盲判 -> /data/pool_blind.jsonl;跑完执行 scripts/pool_blind_report.sh"
