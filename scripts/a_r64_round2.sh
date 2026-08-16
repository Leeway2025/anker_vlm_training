#!/bin/bash
# r64 二轮续训加厚 SubKS 余量(一轮 val 到step200 仍单调下降=欠训)。
# 从一轮 best 权重出发、全新短调度(warmup20/线性、LR减半5e-6),再蒸馏 200 步。
# 一轮锁定版 ns_repair_r64/train_params_best.LOCKED_80p56.npz(80.56)始终保留,
# 二轮只有更好才取代。输出独立目录 ns_repair_r64_r2,评测+公平校准后对比择优。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }
INIT=$M/ns_repair_r64/train_params_best.LOCKED_80p56.npz
O=$M/ns_repair_r64_r2

echo "[$(ts)] r64 二轮续训: init=一轮best  老师=汤u512  LR5e-6  200步"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --train-vision --train-projector \
  --init-npz $INIT \
  --teacher-npz $M/teacher_u512.npz --distill-coef 0.5 --distill-temp 2.0 \
  --augment --accum 16 \
  --lr 5e-6 --proj-lr 1e-5 --vision-lr 5e-6 --loraplus-ratio 1 \
  --warmup 20 --lr-schedule linear \
  --steps 200 --eval-every 50 --early-stop-patience 5 --ckpt-every 100 \
  --seed 11 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && break; sleep 60
done

if [ -f $O/train_params_best.npz ]; then
  echo "[$(ts)] 评测 r64_round2"
  INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json \
    $O/eval_preds $O/train_params_best.npz 8 && \
  python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $O/eval_report.txt
  echo "[$(ts)] === r64_round2 公平校准口径 ==="
  python3 $M/fit_calibration.py $O/eval_preds.jsonl \
    --gold /data/labels_test.jsonl 2>&1 | tee $O/fair_calib.txt
fi
echo "[$(ts)] a_r64_round2 完成"
