#!/bin/bash
# 截断修补(蒸馏轮): 学生=uniform截断产物续训, 老师=满秩汤 pad 到 u512, KL+CE
# 用法: bash scripts/repair_distill.sh <rank> [steps]   如: 72 500
# 前置: outputs/delivery_0807/trunc_r$R/model.npz 与 teacher_u512.npz 已存在
cd /workspace && unset WDS_DIR
R=${1:?用法: repair_distill.sh <rank> [steps]}
STEPS=${2:-500}
M=outputs/delivery_0807
O=$M/repair_r$R
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank $R \
  --train-vision --train-projector \
  --init-npz $M/trunc_r$R/model.npz \
  --teacher-npz $M/teacher_u512.npz --distill-coef 0.5 --distill-temp 2.0 \
  --augment --accum 32 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 50 --lr-schedule linear \
  --steps $STEPS --eval-every 100 --early-stop-patience 4 \
  --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] attempt $ATT $(date)"
  [ $ATT -ge 5 ] && exit 1
  sleep 60
done
INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt && \
python3 $M/apply_calibration.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl | tee $O/calibrated_report.txt
