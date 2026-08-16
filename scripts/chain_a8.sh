#!/bin/bash
# A机:等 soup_all4 推理释放 TPU → 重跑 trunc_tk32(r128/r96 截断评测)
cd /workspace
echo "[chain_a8] 等待 soup_all4 释放 TPU… $(date)"
N=0
while [ ! -f outputs/soup_all4/eval_preds.jsonl ] && [ $N -lt 90 ]; do sleep 60; N=$((N+1)); done
sleep 90
bash scripts/trunc_tk32.sh > outputs/trunc_tk32.log 2>&1
echo "[chain_a8] 完成 $(date)"
