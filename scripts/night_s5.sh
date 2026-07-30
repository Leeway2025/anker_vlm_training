#!/bin/bash
# 夜链 S5: CoT(资产C)从头训(唯一变量) → 测试评测 → 裸logits(独立目录optin_s5)
#          → 手术先验 → 交付候选分;链尾自动补 seed-1 重dump + RT×SK联合解码
# 用法(容器内 /workspace):
#   nohup bash scripts/night_s5.sh > night_s5.log 2>&1 &
# 明早看:
#   outputs/jax_5b_s5/eval_report.txt   (S5 裸分,对 seed-1 裸分 73.52 比)
#   outputs/optin_s5/eval_report.txt    (S5+先验交付候选,对 73.77/RT 83.65 比)
#   outputs/optin/eval_report_joint.txt (seed-1 联合解码,对 83.65 比)
set -e
cd "$(dirname "$0")/.."
echo "[s5] 开跑 $(date) | 代码: $(git log --oneline -1)"
test -s /data/assets_rat/asset_C_reasoning.jsonl || { echo "[s5] 缺 CoT 资产"; exit 1; }

# ① S5 训练: seed-1 配方逐字不动 + CoT 三参数(~5-6h)
python jax_impl/train_sft.py --labels /data/labels_dedup.jsonl \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 3 \
    --accum 32 --steps 1500 --eval-every 100 --val-ids /data/val_ids_v2.txt \
    --seed 1 --mu-dtype float32 \
    --cot-file /data/assets_rat/asset_C_reasoning.jsonl --cot-ratio 0.6 --cot-anneal 0.5 \
    --prefetch-workers 24 --out outputs/jax_5b_s5
echo "[s5] 训完 $(date)"

# ② S5 测试评测(裸分)
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/jax_5b_s5/eval_preds outputs/jax_5b_s5/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/jax_5b_s5/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/jax_5b_s5/eval_report.txt

# ③ S5 裸 logits(独立目录,不覆盖 optin/)→ 手术先验 → 交付候选
mkdir -p outputs/optin_s5
INFER_ARGS='--dump-letter-logits' \
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/optin_s5/preds outputs/jax_5b_s5/train_params_best.npz 8
python3 jax_impl/apply_surgical_prior.py --logits outputs/optin_s5/preds.jsonl \
    --labels /data/labels_test.jsonl \
    --fold-a /data/test_sfoldA.jsonl --fold-b /data/test_sfoldB.jsonl \
    --tau 0.7 --out outputs/optin_s5/preds_surg.jsonl
python3 jax_impl/eval_metrics.py --preds outputs/optin_s5/preds_surg.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/optin_s5/eval_report.txt
echo "[s5] S5 交付候选完成 $(date)"

# ④ 补账: seed-1 裸 logits 重 dump 回 outputs/optin/(修复白天误覆盖)
INFER_ARGS='--dump-letter-logits' \
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/optin/preds outputs/jax_5b_seed1/train_params_best.npz 8

# ⑤ seed-1 联合解码(门禁自检底座;失败不炸整链)
bash scripts/rt_joint_decode.sh || echo "[s5] 联合解码门禁未过,明早人工看"
echo "[s5] 全链完成 $(date)"
