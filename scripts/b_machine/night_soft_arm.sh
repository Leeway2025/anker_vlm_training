#!/bin/bash
# 清洗夜·软化臂(A机): 单变量=RT清洗(softclean+降权), 对基线 seed-1 SubKS 73.52。
# 配方与 seed-1 逐字一致, 仅换 labels + sample-weights;带 cartography+断点。
# 排锁: 等 remat 门跑完自动起。
set -e
cd "$(dirname "$0")/.."
unset WDS_DIR
exec 200>/tmp/night_chain.lock
flock -w 21600 200 || { echo "[FATAL] 等锁 6h 超时"; exit 1; }
ts() { date '+%m-%d %H:%M:%S'; }
OUT=outputs/jax_5b_soft

echo "[$(ts)] 软化臂训练开始"
ATT=0
until python jax_impl/train_sft.py --labels /data/labels_dedup_softclean.jsonl \
    --sample-weights /data/suspect_weights.json \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 3 \
    --accum 32 --steps 1500 --eval-every 100 --val-ids /data/val_ids_v2.txt \
    --seed 1 --mu-dtype float32 --prefetch-workers 24 \
    --cartography --ckpt-every 200 --resume --out $OUT; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && { echo "[watchdog] 连败5次放弃"; exit 1; }
  echo "[watchdog] 非零退出(第${ATT}次), 60s 后断点续跑"; sleep 60
done

echo "[$(ts)] 推理+双口径评测"
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    $OUT/eval_preds $OUT/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $OUT/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class \
    --exclude-ids /data/test_mislabel_exclude_ids_522.txt | tee $OUT/eval_report.txt
echo "[$(ts)] 软化臂完成。验收: SubKS vs 73.52(官方全量块), 不看 test RT"
