#!/bin/bash
# 定型段①:soup_tk32 截断 r128/r96 → 各自 K=32 直评(零重训基线)
set -e
cd "$(dirname "$0")/.."
for R in 128 96; do
  O=outputs/soup_tk32_r$R
  mkdir -p $O
  [ -f $O/model.npz ] || python3 jax_impl/svd_truncate_lora.py \
    --in outputs/soup_tk32/train_params_best.npz --rank $R --out $O/model.npz
  if [ ! -f $O/eval_report.txt ]; then
    SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
      bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
      $O/eval_preds $O/model.npz 8
    python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
      --labels /data/labels_test.jsonl | tee $O/eval_report.txt
  fi
done
echo "[trunc_tk32] 完成 $(date)"
