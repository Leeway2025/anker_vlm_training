#!/bin/bash
# B机:用更好的底子 anneal_c(续退火 best_val 2.2357)做 r64 蒸馏,
#   anneal_c → uniform r64 ASVD 截断 → 汤u512 蒸馏修补 → 评测 + 公平校准。
#   目标:把 SubKS 余量做得比 anneal_b 版(ns_repair_r64=80.56)更厚。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }
BASE=outputs/jax_anneal_c/train_params_best.npz
S=$M/c_single_r64
O=$M/ns_repair_c_r64

# ① ASVD 截断
if [ ! -f $S/model.npz ]; then
  mkdir -p $S
  echo "[$(ts)] ① anneal_c → uniform r64 ASVD 截断"
  python3 jax_impl/svd_truncate_lora.py --in $BASE --rank 64 \
    --act-stats $M/act_stats.npz --out $S/model.npz || exit 1
  echo "anneal_c" > $S/SOURCE
fi

# ② 汤 u512 蒸馏
echo "[$(ts)] ② 蒸馏: 学生=anneal_c_r64 老师=汤u512 200步"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --train-vision --train-projector \
  --init-npz $S/model.npz \
  --teacher-npz $M/teacher_u512.npz --distill-coef 0.5 --distill-temp 2.0 \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear \
  --steps 200 --eval-every 50 --early-stop-patience 4 --ckpt-every 100 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && break; sleep 60
done

# ③ 评测 + 公平校准
if [ -f $O/train_params_best.npz ]; then
  echo "[$(ts)] ③ 评测 ns_repair_c_r64"
  INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json \
    $O/eval_preds $O/train_params_best.npz 8 && \
  python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $O/eval_report.txt
  echo "[$(ts)] === ns_repair_c_r64 公平校准 ==="
  python3 $M/fit_calibration.py $O/eval_preds.jsonl \
    --gold /data/labels_test.jsonl 2>&1 | tee $O/fair_calib.txt
fi
echo "[$(ts)] B机 anneal_c-r64 完成"
