#!/bin/bash
set -e
cd /workspace
mkdir -p outputs/r96v3_int8
python3 outputs/delivery_0807/quantize_lora.py --in outputs/r96v3/train_params_best.npz \
  --out-bf16 outputs/r96v3_int8/model_bf16.npz --out-int8 outputs/r96v3_int8/model_int8sim.npz
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  outputs/r96v3_int8/eval_preds outputs/r96v3_int8/model_int8sim.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/r96v3_int8/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee outputs/r96v3_int8/eval_report.txt
echo "[r96v3_int8] 完成 $(date)"
