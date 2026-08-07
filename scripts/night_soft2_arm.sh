#!/bin/bash
# 过夜臂·软化v2(A机): 单变量=规范重审(对比今晚软化v1臂)。
# 等 assets_v2 就绪 + TPU 锁(排在软化v1臂后)。
set -e
cd "$(dirname "$0")/.."
unset WDS_DIR
while [ ! -f /data/assets_v2.done ]; do sleep 60; done
exec 200>/tmp/night_chain.lock
flock -w 43200 200 || { echo "[FATAL] 等锁 12h 超时"; exit 1; }
ts() { date '+%m-%d %H:%M:%S'; }
OUT=outputs/jax_5b_soft2

echo "[$(ts)] 软化v2臂开训(规范重审后资产)"
ATT=0
until python jax_impl/train_sft.py --labels /data/labels_dedup_softclean_v2.jsonl \
    --sample-weights /data/suspect_weights_v2.json \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 3 \
    --accum 32 --steps 1500 --eval-every 100 --val-ids /data/val_ids_v2.txt \
    --seed 1 --mu-dtype float32 --prefetch-workers 24 \
    --cartography --ckpt-every 200 --resume --out $OUT; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && { echo "[watchdog] 连败5次放弃"; exit 1; }
  echo "[watchdog] 非零退出(第${ATT}次), 60s 后断点续跑"; sleep 60
done

echo "[$(ts)] 推理+双口径评测(v3 口径)"
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    $OUT/eval_preds $OUT/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $OUT/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class \
    --exclude-ids /data/test_mislabel_exclude_ids_rt_v3.txt | tee $OUT/eval_report.txt
echo "[$(ts)] 软化v2臂完成"
