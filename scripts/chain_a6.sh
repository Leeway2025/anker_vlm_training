#!/bin/bash
# A机:等 tksel32b_s2 收官 → 纯选族汤(构汤+推理+报告)
cd /workspace
echo "[chain_a6] 等待 tksel32b_s2 完成… $(date)"
while [ ! -f outputs/tksel32b_s2/eval_report.txt ]; do sleep 60; done
sleep 15
bash scripts/soup_tk32.sh > outputs/soup_tk32.log 2>&1
echo "[chain_a6] 汤完成 $(date)"
