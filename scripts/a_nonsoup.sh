#!/bin/bash
# A机非汤旗舰(用户 0810 拍板:汤不耐压 r128→-0.53 / r96→-0.80,改走单模):
#   单模 anneal_b(现成最好底子) → uniform r64 ASVD激活感知截断(单模谱瘦,损伤小)
#   → 汤 u512 当老师蒸馏(KL+CE,把压缩掉的分从汤里赎回)→ 评测 + 校准。
# 目标:169MB 档守线 RT>=87.91 / SubKS>=80.42(校准口径)。r64 底子成后可再压更小档。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }
SINGLE=outputs/anneal_b_best.npz; SNAME=anneal_b

# ① 单模 r64 ASVD 截断
if [ ! -f $M/ns_single_r64/model.npz ]; then
  mkdir -p $M/ns_single_r64
  echo "[$(ts)] ① $SNAME → uniform r64 ASVD 截断"
  python3 jax_impl/svd_truncate_lora.py --in $SINGLE --rank 64 \
    --act-stats $M/act_stats.npz --out $M/ns_single_r64/model.npz || exit 1
  echo "$SNAME" > $M/ns_single_r64/SOURCE
fi

# ② 汤老师蒸馏修补
echo "[$(ts)] ② 蒸馏修补: 学生=${SNAME}_r64  老师=汤u512  500步"
O=$M/ns_repair_r64; ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --train-vision --train-projector \
  --init-npz $M/ns_single_r64/model.npz \
  --teacher-npz $M/teacher_u512.npz --distill-coef 0.5 --distill-temp 2.0 \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear \
  --steps 200 --eval-every 50 --early-stop-patience 4 --ckpt-every 100 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && break; sleep 60
done

# ③ 评测 + 校准
if [ -f $O/train_params_best.npz ]; then
  echo "[$(ts)] ③ 评测 ns_repair_r64"
  INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json \
    $O/eval_preds $O/train_params_best.npz 8 && \
  python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $O/eval_report.txt && \
  python3 $M/apply_calibration.py $O/eval_preds.jsonl \
    --gold /data/labels_test.jsonl | tee $O/calibrated_report.txt
  gcloud storage cp $O/eval_report.txt $O/calibrated_report.txt \
    gs://zx_vlm_dataset/backup_amachine_0810/ns_repair_r64/ 2>/dev/null || true
fi
echo "[$(ts)] A机非汤旗舰完成: $(tail -1 $O/calibrated_report.txt 2>/dev/null)"
