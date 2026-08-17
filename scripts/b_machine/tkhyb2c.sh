#!/bin/bash
# hyb2b 续退火(0813:best_val恰在1500末步=未收敛;从其best继续,
#   lr 3e-6→1e-6 再退 1000 步,期望再挖 0.1-0.3)。
O=outputs/tkhyb2c
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  TOKEN_COMPRESS_MODE=hyb2 SELECT_TOKENS_K=30 python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --sample-weights /data/sw_rare_700k.json \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/tkhyb2b/train_params_best.npz \
  --lr 1e-6 --vision-lr 3e-6 --proj-lr 7e-5 \
  --augment --accum 32 \
  --steps 1000 --eval-every 100 --early-stop-patience 4 \
  --ckpt-every 400 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[tkhyb2b] 无产物"; exit 1; }

TOKEN_COMPRESS_MODE=hyb2 SELECT_TOKENS_K=30 \
  INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[tkhyb2b] 完成 $(date)"
