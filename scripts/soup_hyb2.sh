#!/bin/bash
# hyb2 族 2 成员汤:avg(tkhyb2b, tkhyb2b_s2) → hyb2 推理 → 报告(A机跑,权重已从B拉来)
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/soup_hyb2
python3 - <<'PY'
import numpy as np
a = np.load('outputs/tkhyb2b/train_params_best.npz')
b = np.load('outputs/tkhyb2b_s2/train_params_best.npz')
out = {k: (0.5*a[k].astype(np.float64) + 0.5*b[k].astype(np.float64)).astype(np.float32)
       for k in a.files if k in b.files and a[k].shape == b[k].shape}
np.savez('outputs/soup_hyb2/train_params_best.npz', **out)
print(f'{len(out)} 键: 0.5*tkhyb2b + 0.5*s2')
PY
TOKEN_COMPRESS_MODE=hyb SELECT_TOKENS_K=30 true  # placeholder避免误读
TOKEN_COMPRESS_MODE=hyb2 SELECT_TOKENS_K=30 INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  outputs/soup_hyb2/eval_preds outputs/soup_hyb2/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/soup_hyb2/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee outputs/soup_hyb2/eval_report.txt
echo "[soup_hyb2] 完成 $(date)"
