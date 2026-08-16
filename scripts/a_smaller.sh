#!/bin/bash
# 更小档蒸馏(r64 已过线 463MB/80.56,试探能压多小仍守 SubKS>=80.42):
#   anneal_b → uniform r{R} ASVD 截断 → 汤 u512 蒸馏修补(与 ns_repair_r64 同超参)
#   → 评测 + 公平重拟校准(fit_calibration)。依次 r48(~386MB)、r32(~287MB)。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }
SINGLE=outputs/anneal_b_best.npz

run_rank() {
  R=$1
  S=$M/ns_single_r$R          # ASVD 底子
  O=$M/ns_repair_r$R          # 蒸馏输出
  # ① ASVD 截断
  if [ ! -f $S/model.npz ]; then
    mkdir -p $S
    echo "[$(ts)] ① anneal_b → uniform r$R ASVD 截断"
    python3 jax_impl/svd_truncate_lora.py --in $SINGLE --rank $R \
      --act-stats $M/act_stats.npz --out $S/model.npz || return 1
    echo "anneal_b" > $S/SOURCE
  fi
  # ② 蒸馏修补
  echo "[$(ts)] ② 蒸馏: 学生=anneal_b_r$R 老师=汤u512 200步"
  ATT=0
  until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
    python jax_impl/train_sft.py $RESUME \
    --labels /data/labels_train_plus_testval_v2.jsonl \
    --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
    --rank-scheme uniform --rank $R \
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
  # ③ 评测 + 公平重拟校准
  if [ -f $O/train_params_best.npz ]; then
    echo "[$(ts)] ③ 评测 r$R"
    INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
      /data/labels_test.jsonl /data/hf_layout.json \
      $O/eval_preds $O/train_params_best.npz 8 && \
    python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
      --labels /data/labels_test.jsonl | tee $O/eval_report.txt
    echo "[$(ts)] === r$R 公平校准口径 ==="
    python3 $M/fit_calibration.py $O/eval_preds.jsonl \
      --gold /data/labels_test.jsonl 2>&1 | tee $O/fair_calib.txt
  fi
}

run_rank 48
run_rank 32
echo "[$(ts)] a_smaller 完成"
