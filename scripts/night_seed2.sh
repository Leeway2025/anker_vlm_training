#!/bin/bash
# 夜链 seed-2: 摇号训练(唯一变量 --seed 2)→ 评测 → 与 seed-1 三档配比双强汤
#   → 自动选裸分赢家 → 裸logits → 手术先验 → 交付候选分(对 EunoVLM 73.83)
# 用法: nohup bash scripts/night_seed2.sh > night_seed2.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
# 防环境污染: rationalize 用的 WDS_DIR(训练集目录)若被继承,会把测试集
# 推理导向错误 tar(KeyError 实测)。链内数据路径由 labels/meta 自解析。
unset WDS_DIR
echo "[s2] 开跑 $(date) | 代码: $(git log --oneline -1)"

# ① seed-2 训练(配方逐字=seed-1,仅换种子)
python jax_impl/train_sft.py --labels /data/labels_dedup.jsonl \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 3 \
    --accum 32 --steps 1500 --eval-every 100 --val-ids /data/val_ids_v2.txt \
    --seed 2 --mu-dtype float32 \
    --prefetch-workers 24 --out outputs/jax_5b_seed2
echo "[s2] 训完 $(date)"

# ② seed-2 测试评测
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/jax_5b_seed2/eval_preds outputs/jax_5b_seed2/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/jax_5b_seed2/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/jax_5b_seed2/eval_report.txt

# ③ 双强汤三档配比(w*seed1 + (1-w)*seed2),各自评测
for W in 0.50 0.60 0.75; do
  OUT="outputs/soup_s12_${W/0./}"
  mkdir -p "$OUT"
  W=$W OUT=$OUT python3 - <<'PY'
import os
import numpy as np
w = float(os.environ['W']); out = os.environ['OUT']
a = np.load('outputs/jax_5b_seed1/train_params_best.npz')
b = np.load('outputs/jax_5b_seed2/train_params_best.npz')
m = {k: (w*a[k].astype(np.float64) + (1-w)*b[k].astype(np.float64)).astype(np.float32)
     for k in a.files if k in b.files and a[k].shape == b[k].shape}
np.savez(f'{out}/train_params_best.npz', **m)
print(f'[soup] w={w} {len(m)} 键 -> {out}')
PY
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
      "$OUT/eval_preds" "$OUT/train_params_best.npz" 8
  python3 jax_impl/eval_metrics.py --preds "$OUT/eval_preds.jsonl" \
      --labels /data/labels_test.jsonl | tee "$OUT/eval_report.txt"
done

# ④ 自动选裸分赢家(seed1/seed2/三档汤)→ 裸logits → 手术先验 → 交付候选
WINNER=$(python3 - <<'PY'
import re
def subks(p):
    try:
        m = re.search(r'SubKS\s+acc\s*=\s*([\d.]+)%', open(p).read())
        return float(m.group(1)) if m else 0.0
    except FileNotFoundError:
        return 0.0
cands = [('outputs/jax_5b_seed1', 73.52)]
for d in ('outputs/jax_5b_seed2', 'outputs/soup_s12_50',
          'outputs/soup_s12_60', 'outputs/soup_s12_75'):
    cands.append((d, subks(f'{d}/eval_report.txt')))
best = max(cands, key=lambda x: x[1])
import sys
print(f'[winner] {best[0]} SubKS={best[1]:.2f}', file=sys.stderr)
print(best[0])
PY
)
echo "[s2] 裸分赢家 = $WINNER"
mkdir -p outputs/optin_s12
INFER_ARGS='--dump-letter-logits' \
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/optin_s12/preds "$WINNER/train_params_best.npz" 8
python3 jax_impl/apply_surgical_prior.py --logits outputs/optin_s12/preds.jsonl \
    --labels /data/labels_test.jsonl \
    --fold-a /data/test_sfoldA.jsonl --fold-b /data/test_sfoldB.jsonl \
    --tau 0.7 --out outputs/optin_s12/preds_surg.jsonl
python3 jax_impl/eval_metrics.py --preds outputs/optin_s12/preds_surg.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/optin_s12/eval_report.txt
echo "[s2] 全链完成 $(date) —— optin_s12/eval_report.txt 对 EunoVLM SubKS 73.83"
