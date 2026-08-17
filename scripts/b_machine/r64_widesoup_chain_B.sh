#!/bin/bash
# B机: 更宽满token汤(soupw1+soupw3)→ SVD r64 → int8 → 满token int8评测 → 严格OOF
# (与A互补: A出bf16读数, B出int8交付数)
set -e
cd /workspace
unset WDS_DIR
OUT=outputs/r64_widesoup
mkdir -p "$OUT"
echo "[widesoupB] start $(date)"

python3 - <<'PY'
import numpy as np
a=np.load('outputs/soupw1/soupw1.npz'); b=np.load('outputs/soupw3/soupw3.npz')
keys=[k for k in a.files if k in b.files and a[k].shape==b[k].shape]
m={k:(0.5*a[k].astype(np.float64)+0.5*b[k].astype(np.float64)).astype(np.float32) for k in keys}
for k in a.files:
    if k not in m: m[k]=a[k]
np.savez('outputs/r64_widesoup/soup_ft.npz', **m)
print(f"[widesoupB] averaged {len(keys)} matched, total {len(m)}")
PY

python3 jax_impl/svd_truncate_lora.py --in outputs/r64_widesoup/soup_ft.npz \
    --rank 64 --out "$OUT/model.npz" 2>&1 | tail -10
echo "[widesoupB] truncated $(date)"

python3 outputs/delivery_0807/quantize_lora.py --in "$OUT/model.npz" \
    --out-int8 "$OUT/model_int8_sim.npz" 2>&1 | tail -10
echo "[widesoupB] int8 quant done $(date)"

bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    "$OUT/eval_preds_int8" "$OUT/model_int8_sim.npz" 8 2>&1 | tail -12
echo "[widesoupB] int8 eval done $(date)"

echo "===== OOF int8 满 token ====="
python3 outputs/class_diag_affine.py "$OUT/eval_preds_int8.jsonl" --gold /data/labels_test.jsonl --folds 5
echo "[widesoupB] ALL DONE $(date)"
