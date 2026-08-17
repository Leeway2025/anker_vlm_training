#!/bin/bash
# 1M 全量训练 v1(P5 冒烟放行后使用;2 epoch 约 6 天,需长机时窗口)
# 断点续训已内置: --ckpt-every 250 全量落盘(参数+优化器+进度,滚动 2 份),
# watchdog 外壳自动续跑 —— 抢占/OOM/网络抖动都从最近断点爬起,无人值守。
# 用法: nohup bash scripts/train_1m.sh > train_1m.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
unset WDS_DIR

ATT=0
until python jax_impl/train_sft.py --labels /data/labels_1m_v1.jsonl \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 4 \
    --accum 32 --steps 8000 --eval-every 250 --val-ids /data/val_ids_1m.txt \
    --seed 1 --mu-dtype float32 --prefetch-workers 24 \
    --ckpt-every 250 --resume --out outputs/jax_1m_v1; do
  ATT=$((ATT+1))
  echo "[watchdog] 训练退出非零(第 ${ATT} 次)$(date '+%m-%d %H:%M') —— 60s 冷却后从断点续跑" >&2
  if [ "$ATT" -ge 20 ]; then
    echo "[watchdog] 连续失败 20 次,放弃 —— 查 train_1m.log 尾部与 outputs/jax_1m_v1/ckpt_latest.npz" >&2
    exit 1
  fi
  sleep 60   # TPU 进程退出后 libtpu 锁需 >10s 才释放,60s 稳妥
done

bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/jax_1m_v1/eval_preds outputs/jax_1m_v1/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/jax_1m_v1/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/jax_1m_v1/eval_report.txt
