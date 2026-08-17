#!/bin/bash
set -e
cd "$(dirname "$0")/.."
unset WDS_DIR
exec 200>/tmp/night_chain.lock
flock -w 43200 200 || exit 1
ts() { date '+%m-%d %H:%M'; }
echo "[$(ts)] P5冒烟: 自然100k, 对73.52±0.5"
OUT=outputs/jax_p5_smoke
ATT=0
until python jax_impl/train_sft.py --labels /data/pool_natural100k_labels.jsonl \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 3 \
    --accum 32 --steps 1500 --eval-every 100 --val-n 1657 \
    --seed 1 --mu-dtype float32 --prefetch-workers 24 \
    --cartography --ckpt-every 200 --resume --out $OUT; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && exit 1; sleep 60
done
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    $OUT/eval_preds $OUT/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $OUT/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $OUT/eval_report.txt
echo "[$(ts)] 冒烟完, 接1M全池打分(~13h)"
mkdir -p outputs/gate2_full
INFER_ARGS='--dump-letter-logits --max-new 4' bash jax_impl/infer_sharded.sh \
    python /data/pool_full_labels.jsonl /data/hf_layout.json \
    outputs/gate2_full/scores outputs/jax_5b_seed1/train_params_best.npz 8
n=$(wc -l < outputs/gate2_full/scores.jsonl 2>/dev/null || echo 0)
echo "[$(ts)] 1M打分产出 $n 行"
