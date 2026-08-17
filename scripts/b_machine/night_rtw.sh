#!/bin/bash
# 夜链 ④: RT 字母位加权 x8(唯一变量,seed-1 配方逐字不动,无 CoT)
#   → 测试评测 → 裸logits(optin_rtw) → 手术先验 → 交付候选分
# 用法: nohup bash scripts/night_rtw.sh > night_rtw.log 2>&1 &
# 明早对照: 裸分对 seed-1 73.52/RT~83.4;先验版对交付 73.77/RT 83.65
# 验收: RT >= +0.3 且 SubKS 回撤 <= 0.2 才上位;否则回退 seed-1
set -e
cd "$(dirname "$0")/.."
# 防环境污染: rationalize 用的 WDS_DIR(训练集目录)若被继承,会把测试集
# 推理导向错误 tar(KeyError 实测)。链内数据路径由 labels/meta 自解析。
unset WDS_DIR
echo "[rtw] 开跑 $(date) | 代码: $(git log --oneline -1)"

python jax_impl/train_sft.py --labels /data/labels_dedup.jsonl \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 3 \
    --accum 32 --steps 1500 --eval-every 100 --val-ids /data/val_ids_v2.txt \
    --seed 1 --mu-dtype float32 --rt-w 8 \
    --prefetch-workers 24 --out outputs/jax_5b_rtw
echo "[rtw] 训完 $(date)"

bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/jax_5b_rtw/eval_preds outputs/jax_5b_rtw/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/jax_5b_rtw/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/jax_5b_rtw/eval_report.txt

mkdir -p outputs/optin_rtw
INFER_ARGS='--dump-letter-logits' \
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/optin_rtw/preds outputs/jax_5b_rtw/train_params_best.npz 8
python3 jax_impl/apply_surgical_prior.py --logits outputs/optin_rtw/preds.jsonl \
    --labels /data/labels_test.jsonl \
    --fold-a /data/test_sfoldA.jsonl --fold-b /data/test_sfoldB.jsonl \
    --tau 0.7 --out outputs/optin_rtw/preds_surg.jsonl
python3 jax_impl/eval_metrics.py --preds outputs/optin_rtw/preds_surg.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/optin_rtw/eval_report.txt
echo "[rtw] 全链完成 $(date)"
