#!/bin/bash
cd /workspace
echo "[chain_a11] 等 vr_k32_zero 探针收官… $(date)"
N=0
while [ ! -f outputs/vr_k32_zero/eval_report.txt ] && [ $N -lt 40 ]; do sleep 60; N=$((N+1)); done
sleep 30
bash scripts/vrk32_distill.sh > outputs/vrk32_distill.log 2>&1
echo "[chain_a11] 完成 $(date)"
