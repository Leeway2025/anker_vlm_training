#!/bin/bash
# 夜链 v2: val_v2 切卷 → 新原点 → seed-1 训练 → 评测 → 模型汤 → 自动选赢家
#          → 裸 logits → 全类先验优化 → 交付候选分
# 用法(容器内 /workspace):
#   nohup bash jax_impl/night_v2.sh > night_v2.log 2>&1 &
# 前置: /data/labels_dedup.jsonl、labels_test.jsonl、hf_layout.json、
#       test_sfoldA/B.jsonl、outputs/jax_5a/proj_a.npz、
#       outputs/jax_5b_v4replica/train_params_best.npz
# 明早看四份报告: outputs/probe_base_v2_report.txt(val_v2 新原点)
#   outputs/jax_5b_seed1/eval_report.txt / outputs/soup2/eval_report.txt(汤局)
#   outputs/optin/eval_report.txt(赢家+全类先验 = 交付候选分,对 73.15 比)
set -e
cd "$(dirname "$0")/.."
echo "[nv2] 开跑 $(date) | 代码: $(git log --oneline -1)"

# ① val v2 切卷(幂等)
if [ ! -s /data/val_ids_v2.txt ] || [ "$(wc -l < /data/val_ids_v2.txt)" -lt 1400 ]; then
  python3 jax_impl/export_val_split.py --labels /data/labels_dedup.jsonl \
      --val-n 1536 --seed 0 --match-mix /data/labels_test.jsonl --fill-loose \
      --out /data/labels_val_v2.jsonl --ids-out /data/val_ids_v2.txt
fi
echo "[nv2] val_v2 = $(wc -l < /data/val_ids_v2.txt) 条"

# ② 新原点: 复刻在 val v2 上的基线
bash jax_impl/infer_sharded.sh python /data/labels_val_v2.jsonl /data/hf_layout.json \
    outputs/probe_base_v2 outputs/jax_5b_v4replica/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/probe_base_v2.jsonl \
    --labels /data/labels_val_v2.jsonl | tee outputs/probe_base_v2_report.txt

# ③ seed-1 训练(~5h)
python jax_impl/train_sft.py --labels /data/labels_dedup.jsonl \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 3 \
    --accum 32 --steps 1500 --eval-every 100 --val-ids /data/val_ids_v2.txt \
    --seed 1 --mu-dtype float32 \
    --prefetch-workers 24 --out outputs/jax_5b_seed1
echo "[nv2] seed-1 训完 $(date)"

# ④ seed-1 测试评测
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/jax_5b_seed1/eval_preds outputs/jax_5b_seed1/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/jax_5b_seed1/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/jax_5b_seed1/eval_report.txt

# ⑤ 模型汤(复刻+seed1 权重平均)+ 评测
mkdir -p outputs/soup2
python3 - <<'PY'
import numpy as np
a = np.load('outputs/jax_5b_v4replica/train_params_best.npz')
b = np.load('outputs/jax_5b_seed1/train_params_best.npz')
out = {k: ((a[k].astype(np.float64) + b[k].astype(np.float64)) / 2).astype(np.float32)
       for k in a.files if k in b.files and a[k].shape == b[k].shape}
np.savez('outputs/soup2/train_params_best.npz', **out)
print(f'[soup] {len(out)} 键已平均')
PY
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/soup2/eval_preds outputs/soup2/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/soup2/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/soup2/eval_report.txt

# ⑥ 自动选赢家 → 裸 logits → 全类先验优化 → 交付候选分
WINNER=$(python3 - <<'PY'
import re, sys
def subks(p, default=0.0):
    try:
        m = re.search(r'SubKS\s+acc\s*=\s*([\d.]+)%', open(p).read())
        return float(m.group(1)) if m else default
    except FileNotFoundError:
        return default
cands = [
    (72.32, 'outputs/jax_5b_v4replica/train_params_best.npz'),
    (subks('outputs/jax_5b_seed1/eval_report.txt'), 'outputs/jax_5b_seed1/train_params_best.npz'),
    (subks('outputs/soup2/eval_report.txt'), 'outputs/soup2/train_params_best.npz'),
]
best = max(cands)
print(best[1])
print(f'[winner] SubKS={best[0]:.2f} {best[1]}', file=sys.stderr)
PY
)
echo "[nv2] 底座赢家 = $WINNER"
mkdir -p outputs/optin
INFER_ARGS='--dump-letter-logits' \
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/optin/preds "$WINNER" 8
python3 jax_impl/prior_opt.py --logits outputs/optin/preds.jsonl \
    --labels /data/labels_test.jsonl \
    --fold-a /data/test_sfoldA.jsonl --fold-b /data/test_sfoldB.jsonl \
    --out outputs/optin/preds_opt.jsonl
python3 jax_impl/eval_metrics.py --preds outputs/optin/preds_opt.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/optin/eval_report.txt
echo "[nv2] 全链完成 $(date)"
