#!/bin/bash
# hyb2 升级臂(B机,0812 用户确认排队;与 hyb30 严格同预算 514):
#   ①打分参照帧0(补"到场后静止"盲区)②摘要相似度加权 ③落选者聚 4 个
#   独立摘要 token(选中前景零污染)。先零样本(~25min)再适配重训。
cd /workspace && unset WDS_DIR

Z=outputs/tkhyb2_zero
if [ ! -f $Z/eval_report.txt ]; then
  mkdir -p $Z
  TOKEN_COMPRESS_MODE=hyb2 SELECT_TOKENS_K=30 \
    INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
    bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json \
    $Z/eval_preds outputs/soupw1/soupw1.npz 8 && \
  python3 jax_impl/eval_metrics.py --preds $Z/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $Z/eval_report.txt
fi

O=outputs/tkhyb2_adapt
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  TOKEN_COMPRESS_MODE=hyb2 SELECT_TOKENS_K=30 python jax_impl/train_sft.py $RESUME \
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
[ -f $O/train_params_best.npz ] || { echo "[tkhyb2] 无产物"; exit 1; }

TOKEN_COMPRESS_MODE=hyb2 SELECT_TOKENS_K=30 \
  INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[tkhyb2_adapt] 完成 $(date)"
