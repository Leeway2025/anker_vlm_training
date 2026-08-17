#!/bin/bash
# B机夜班第二棒(单模→压缩→蒸馏路线, 用户 0810 拍板):
#   等 night_soup 完 → git pull → 等 A机物料(teacher_u512 + act_stats)
#   → 选今晚最佳单模(anneal_c 胜过 anneal_b 用 c, 否则 b)
#   → uniform r64 激活感知截断(学生底子, 谱瘦损伤小)
#   → 蒸馏修补(老师=汤u512, KL+CE, 400步)→ 评测。
# 目标: 169MB 档守线 RT>=87.91 / SubKS>=80.42(校准口径)。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }

while ! grep -q 'B机夜班完成' logs/night_soup.log 2>/dev/null; do sleep 300; done
sleep 30
git pull --ff-only 2>&1 | tail -1

while [ ! -f $M/.staged_ok ]; do
  echo "[$(ts)] 等 A机 staging(teacher_u512 + act_stats)"; sleep 300
done

# 选最佳单模: anneal_c 裸 SubKS > anneal_b(79.34)则用 c
SINGLE=outputs/anneal_b_best.npz; SNAME=anneal_b
CSK=$(grep -oP 'SubKS\s+acc\s+=\s+\K[0-9.]+' outputs/jax_anneal_c/eval_report.txt 2>/dev/null)
if [ -n "$CSK" ] && python3 -c "exit(0 if float('$CSK') > 79.34 else 1)"; then
  SINGLE=outputs/jax_anneal_c/train_params_best.npz; SNAME=anneal_c
fi
echo "[$(ts)] 学生底子 = $SNAME(anneal_c SubKS=${CSK:-无})"

if [ ! -f $M/single_r64/model.npz ]; then
  mkdir -p $M/single_r64
  python3 jax_impl/svd_truncate_lora.py --in $SINGLE --rank 64 \
    --act-stats $M/act_stats.npz --out $M/single_r64/model.npz || exit 1
  echo "$SNAME" > $M/single_r64/SOURCE
fi

echo "[$(ts)] 蒸馏修补: 学生=${SNAME}_r64 老师=汤u512, 400步"
O=$M/repair_single_r64; ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --train-vision --train-projector \
  --init-npz $M/single_r64/model.npz \
  --teacher-npz $M/teacher_u512.npz --distill-coef 0.5 --distill-temp 2.0 \
  --augment --accum 32 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 50 --lr-schedule linear \
  --steps 400 --eval-every 100 --early-stop-patience 4 --ckpt-every 200 \
  --seed 11 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && break; sleep 60
done
if [ -f $O/train_params_best.npz ]; then
  INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json \
    $O/eval_preds $O/train_params_best.npz 8 && \
  python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $O/eval_report.txt && \
  python3 $M/apply_calibration.py $O/eval_preds.jsonl \
    --gold /data/labels_test.jsonl | tee $O/calibrated_report.txt
fi
echo "[$(ts)] B机修补棒完成"
