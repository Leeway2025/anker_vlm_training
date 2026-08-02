#!/bin/bash
# 1M 全量训练 v1(草案配方,P5 冒烟放行后使用;预计 25-35h,需长机时窗口)
# ⚠ 断点续训(checkpoint+optimizer落盘)上线前,勿在会关机的窗口启动
# 用法: nohup bash scripts/train_1m.sh > train_1m.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
unset WDS_DIR
python jax_impl/train_sft.py --labels /data/labels_1m_v1.jsonl \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 4 \
    --accum 32 --steps 8000 --eval-every 250 --val-ids /data/val_ids_1m.txt \
    --seed 1 --mu-dtype float32 --prefetch-workers 24 --out outputs/jax_1m_v1
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/jax_1m_v1/eval_preds outputs/jax_1m_v1/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/jax_1m_v1/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/jax_1m_v1/eval_report.txt
