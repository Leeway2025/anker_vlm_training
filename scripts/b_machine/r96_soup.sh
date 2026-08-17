#!/bin/bash
set -e
cd /workspace
mkdir -p outputs/r96_soup
python3 - <<'PY'
import numpy as np
a = np.load("outputs/r96_recover/train_params_best.npz")
b = np.load("outputs/r96v3/train_params_best.npz")
out = {k: (0.5*a[k].astype(np.float64)+0.5*b[k].astype(np.float64)).astype(np.float32)
       for k in a.files if k in b.files and a[k].shape==b[k].shape}
np.savez("outputs/r96_soup/train_params_best.npz", **out)
print(len(out), "keys souped")
PY
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  outputs/r96_soup/eval_preds outputs/r96_soup/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/r96_soup/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee outputs/r96_soup/eval_report.txt
echo "[r96_soup] 完成 $(date)"
