#!/bin/bash
# 1M P5: 管线冒烟 = 从 1M v1 池自然抽 10 万训一发(v4配方逐字),对照现100k底座
#   回答: ①管线正确性 ②新池单位数据质量。~7h 过夜。
# 用法: nohup bash scripts/smoke_1m.sh > smoke_1m.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
unset WDS_DIR
test -s /data/labels_1m_v1.jsonl || { echo "先跑 build_pool_v1.sh"; exit 1; }

python3 - <<'PY'
import json, random
random.seed(0)
rows = open('/data/labels_1m_v1.jsonl', encoding='utf-8').readlines()
val = {x.strip() for x in open('/data/val_ids_1m.txt')}
pool = [l for l in rows if json.loads(l)['video_id'] not in val]
random.shuffle(pool)
keep = pool[:100000 - len(val)] + [l for l in rows if json.loads(l)['video_id'] in val]
open('/data/labels_1m_smoke100k.jsonl', 'w', encoding='utf-8').writelines(keep)
print(f'[smoke] 抽样 {len(keep)}(含val卷)-> labels_1m_smoke100k.jsonl')
PY

python jax_impl/train_sft.py --labels /data/labels_1m_smoke100k.jsonl \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 3 \
    --accum 32 --steps 1500 --eval-every 100 --val-ids /data/val_ids_1m.txt \
    --seed 1 --mu-dtype float32 --prefetch-workers 24 --out outputs/jax_1m_smoke
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/jax_1m_smoke/eval_preds outputs/jax_1m_smoke/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/jax_1m_smoke/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/jax_1m_smoke/eval_report.txt
echo "[smoke] 判读: 对 seed-1 裸分 73.52 比 —— ±0.5 内=新池质量同级,管线放行;低 1 分以上=池有系统性问题,回 P2 报告找病类"
