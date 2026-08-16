#!/bin/bash
# A机:等 8f 探针等占卡任务空闲 → r96 短恢复(撞忙重试)
cd /workspace
echo "[chain_a10] 等待 TPU 空闲… $(date)"
sleep 60
for T in 1 2 3 4 5 6 7 8; do
  BUSY=$(ls /proc/*/cmdline 2>/dev/null | xargs -I{} sh -c "tr '\0' ' ' < {} 2>/dev/null" 2>/dev/null | grep -cE "infer\.py|train_sft" || true)
  [ "$BUSY" = "0" ] && break
  sleep 180
done
for T in 1 2 3 4 5 6; do
  bash scripts/r96_recover.sh > outputs/r96_recover.log 2>&1 && break
  echo "[chain_a10] 撞忙重试 $T $(date)"; sleep 300
done
echo "[chain_a10] 完成 $(date)"
