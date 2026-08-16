#!/bin/bash
set -e
cd /workspace
unset WDS_DIR
OUT=outputs/r64_widesoup
echo "[evalA] start $(date)"
INFER_ARGS='--dump-letter-logits' bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    "$OUT/eval_preds" "$OUT/model.npz" 8 2>&1 | tail -12
echo "[evalA] eval done $(date)"
echo "===== OOF bf16 满 token (widesoup r64) ====="
python3 outputs/class_diag_affine.py "$OUT/eval_preds.jsonl" --gold /data/labels_test.jsonl --folds 5
echo "[evalA] ALL DONE $(date)"
