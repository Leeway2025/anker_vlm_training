#!/bin/bash
# A机:等 soup_tk32 完成 → hyb2 族汤
cd /workspace
echo "[chain_a7] 等待 soup_tk32 完成… $(date)"
while [ ! -f outputs/soup_tk32/eval_report.txt ]; do sleep 60; done
sleep 15
bash scripts/soup_hyb2.sh > outputs/soup_hyb2.log 2>&1
echo "[chain_a7] 完成 $(date)"
