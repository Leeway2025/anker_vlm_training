#!/bin/bash
# 训练无关: 更宽的满 token 汤(soupw1+soupw3 meta-soup)→ SVD 截断 r64 → 满 token 评测 → 严格 OOF
# 目的: 给"r64 必须过"一次诚实机会。汤=提升泛化(OOF),非过拟合。
set -e
cd /workspace
unset WDS_DIR
OUT=outputs/r64_widesoup
mkdir -p "$OUT"
echo "[widesoup] start $(date)"

# 1) meta-soup: 等权平均 soupw1 + soupw3 (匹配键/形状)
python3 - <<'PY'
import numpy as np
a=np.load('outputs/soupw1/soupw1.npz'); b=np.load('outputs/soupw3/soupw3.npz')
keys=[k for k in a.files if k in b.files and a[k].shape==b[k].shape]
m={k:(0.5*a[k].astype(np.float64)+0.5*b[k].astype(np.float64)).astype(np.float32) for k in keys}
# 透传 a 中 b 缺失/形状不符的键(保完整)
for k in a.files:
    if k not in m: m[k]=a[k]
np.savez('outputs/r64_widesoup/soup_ft.npz', **m)
print(f"[widesoup] averaged {len(keys)} matched keys, total {len(m)} -> soup_ft.npz")
PY

# 2) SVD 截断到 r64 uniform (纯 SVD, 与 r128 交付同法)
python3 jax_impl/svd_truncate_lora.py --in outputs/r64_widesoup/soup_ft.npz \
    --rank 64 --out "$OUT/model.npz" 2>&1 | tail -20
echo "[widesoup] truncated to r64 $(date)"

# 3) 满 token 评测 (无 SELECT_TOKENS_K => 全 1024 token, bf16)
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    "$OUT/eval_preds" "$OUT/model.npz" 8 2>&1 | tail -15
echo "[widesoup] eval done $(date)"

# 4) 严格 OOF (bf16 满 token)
echo "===== OOF bf16 满 token ====="
python3 outputs/class_diag_affine.py "$OUT/eval_preds.jsonl" --gold /data/labels_test.jsonl --folds 5
echo "[widesoup] ALL DONE $(date)"
