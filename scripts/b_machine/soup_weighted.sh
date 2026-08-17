#!/bin/bash
# 加权模型汤: 0.75*seed1 + 0.25*replica → 测试集评测
# 用法: bash scripts/soup_weighted.sh   (容器内 /workspace)
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/soup_w
python3 - <<'PY'
import numpy as np
a = np.load('outputs/jax_5b_seed1/train_params_best.npz')
b = np.load('outputs/jax_5b_v4replica/train_params_best.npz')
out = {k: (0.75*a[k].astype(np.float64) + 0.25*b[k].astype(np.float64)).astype(np.float32)
       for k in a.files if k in b.files and a[k].shape == b[k].shape}
np.savez('outputs/soup_w/train_params_best.npz', **out)
print(f'{len(out)} 键: 0.75*seed1 + 0.25*replica')
PY
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/soup_w/eval_preds outputs/soup_w/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/soup_w/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/soup_w/eval_report.txt
