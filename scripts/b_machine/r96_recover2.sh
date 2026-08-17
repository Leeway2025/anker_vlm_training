#!/bin/bash
# r96 强恢复 v2(B机,0813:v1踩线未过——uniform差RT0.03/dyn差SubKS0.20):
#   在v1基础上 KD 500→800 步 + --rt-w 2.0(RT字母位加权,专补RT轴)。
export SELECT_TOKENS_K=32
O=outputs/r96_recover2
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 96 \
  --init-npz outputs/soup_tk32_r96/model.npz \
  --teacher-npz outputs/soup_size/teacher_u512.npz \
  --distill-coef 0.5 --distill-temp 2.0 --rt-w 2.0 \
  --train-vision --train-projector \
  --sample-weights /data/sw_rare_700k.json \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 \
  --loraplus-ratio 1 --warmup 30 --lr-schedule linear \
  --steps 800 --eval-every 50 --early-stop-patience 4 --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[r96_recover] 无产物"; exit 1; }
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[r96_recover] 完成 $(date)"
