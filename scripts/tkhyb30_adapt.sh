#!/bin/bash
# Token 压缩 hyb 适配臂(A机;用户 0812 "2+3+4 一起最合理"):
#   TOKEN_COMPRESS_MODE=hyb SELECT_TOKENS_K=30 → 帧0全保64 + 帧1..15各Top-30
#   + 非选中就近合并(总514≈K=32纯选择臂的512,严格同预算可比)。
#   余配方与 tksel32_adapt 完全一致(soupw1 暖启,700k池 v1增强,prod秩)。
cd /workspace && unset WDS_DIR
O=outputs/tkhyb30_adapt
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  TOKEN_COMPRESS_MODE=hyb SELECT_TOKENS_K=30 python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/soupw1/soupw1.npz \
  --augment --accum 32 \
  --steps 1500 --eval-every 100 --early-stop-patience 3 \
  --ckpt-every 400 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[tkhyb30] 无产物"; exit 1; }

TOKEN_COMPRESS_MODE=hyb SELECT_TOKENS_K=30 \
  INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[tkhyb30_adapt] 完成 $(date)"
