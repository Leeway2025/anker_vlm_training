#!/bin/bash
# tkhyb2b 第二 seed 成员(0812 用户"跑完自行往目标卷":压缩域汤原料,
#   与 tkhyb2b 唯一差异 seed 7→11)。
O=outputs/tkhyb2b_s2
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  TOKEN_COMPRESS_MODE=hyb2 SELECT_TOKENS_K=30 python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --sample-weights /data/sw_rare_700k.json \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/soupw1/soupw1.npz \
  --lr 3e-6 --vision-lr 8e-6 --proj-lr 2e-4 \
  --augment --accum 32 \
  --steps 1500 --eval-every 100 --early-stop-patience 4 \
  --ckpt-every 400 \
  --seed 11 --mu-dtype float32 --prefetch-workers 24 \
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
