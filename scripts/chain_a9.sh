#!/bin/bash
# A机:等并行会话的 soup_size 流水线空闲(train.log 5分钟无更新且无新推理)
#   → 补跑 trunc_tk32(r128/r96 阶梯直评),撞忙自动重试。
cd /workspace
echo "[chain_a9] 等待 TPU 空闲… $(date)"
while true; do
  BUSY=0
  for f in outputs/soup_size/distill/train.log outputs/soup_size/*.log; do
    [ -n "$(find $f -mmin -5 2>/dev/null)" ] && BUSY=1
  done
  [ $BUSY -eq 0 ] && break
  sleep 120
done
echo "[chain_a9] 空闲,开始截断评测 $(date)"
for T in 1 2 3 4 5 6; do
  bash scripts/trunc_tk32.sh > outputs/trunc_tk32.log 2>&1 && break
  echo "[chain_a9] 撞忙重试 $T $(date)"; sleep 300
done
echo "[chain_a9] 完成 $(date)"
