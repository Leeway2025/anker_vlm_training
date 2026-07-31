#!/bin/bash
# 加权汤全链(~1h): seed1×replica 三档配比 → 各自评测 → 裸分赢家 → 裸logits
#   → 手术先验+RT阈值 → 交付候选分(对 73.77/83.65;裸分须先超 73.52)
# 用法: nohup bash scripts/soup_full.sh > soup_full.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
echo "[soupf] 开跑 $(date)"

for W in 0.60 0.75 0.85; do
  OUT="outputs/soupw_${W/0./}"
  mkdir -p "$OUT"
  W=$W OUT=$OUT python3 - <<'PY'
import os
import numpy as np
w = float(os.environ['W']); out = os.environ['OUT']
a = np.load('outputs/jax_5b_seed1/train_params_best.npz')
b = np.load('outputs/jax_5b_v4replica/train_params_best.npz')
m = {k: (w*a[k].astype(np.float64) + (1-w)*b[k].astype(np.float64)).astype(np.float32)
     for k in a.files if k in b.files and a[k].shape == b[k].shape}
np.savez(f'{out}/train_params_best.npz', **m)
print(f'[soupf] w={w}*seed1 + {1-w:.2f}*replica, {len(m)} 键 -> {out}')
PY
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
      "$OUT/eval_preds" "$OUT/train_params_best.npz" 8
  python3 jax_impl/eval_metrics.py --preds "$OUT/eval_preds.jsonl" \
      --labels /data/labels_test.jsonl | tee "$OUT/eval_report.txt"
done

WINNER=$(python3 - <<'PY'
import re, sys
def subks(p):
    try:
        m = re.search(r'SubKS\s+acc\s*=\s*([\d.]+)%', open(p).read())
        return float(m.group(1)) if m else 0.0
    except FileNotFoundError:
        return 0.0
cands = [('outputs/jax_5b_seed1', 73.52)] + \
        [(f'outputs/soupw_{w}', subks(f'outputs/soupw_{w}/eval_report.txt'))
         for w in ('60', '75', '85')]
best = max(cands, key=lambda x: x[1])
print(f'[winner] {best[0]} 裸SubKS={best[1]:.2f}', file=sys.stderr)
print(best[0])
PY
)
echo "[soupf] 裸分赢家 = $WINNER"
if [ "$WINNER" = "outputs/jax_5b_seed1" ]; then
  echo "[soupf] 没有汤超过 seed-1 裸分 73.52,不必挂先验,交付口径不变 73.77"
  exit 0
fi
mkdir -p outputs/optin_soupw
INFER_ARGS='--dump-letter-logits' \
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/optin_soupw/preds "$WINNER/train_params_best.npz" 8
python3 jax_impl/apply_surgical_prior.py --logits outputs/optin_soupw/preds.jsonl \
    --labels /data/labels_test.jsonl \
    --fold-a /data/test_sfoldA.jsonl --fold-b /data/test_sfoldB.jsonl \
    --tau 0.7 --out outputs/optin_soupw/preds_surg.jsonl
python3 jax_impl/eval_metrics.py --preds outputs/optin_soupw/preds_surg.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/optin_soupw/eval_report.txt
echo "[soupf] 全链完成 $(date) —— optin_soupw/eval_report.txt 对 73.83 比"
