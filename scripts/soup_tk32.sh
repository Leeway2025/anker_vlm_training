#!/bin/bash
# 纯选族 2 成员汤:avg(tksel32b, tksel32b_s2) → K=32 推理 → 报告
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/soup_tk32
python3 - <<'PY'
import numpy as np
a = np.load('outputs/tksel32b/train_params_best.npz')
b = np.load('outputs/tksel32b_s2/train_params_best.npz')
out = {k: (0.5*a[k].astype(np.float64) + 0.5*b[k].astype(np.float64)).astype(np.float32)
       for k in a.files if k in b.files and a[k].shape == b[k].shape}
np.savez('outputs/soup_tk32/train_params_best.npz', **out)
print(f'{len(out)} 键: 0.5*tksel32b + 0.5*s2')
PY
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  outputs/soup_tk32/eval_preds outputs/soup_tk32/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/soup_tk32/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee outputs/soup_tk32/eval_report.txt
echo "[soup_tk32] 完成 $(date)"
