#!/bin/bash
# 链式排满 · A机第二棒(0812 改排,替代 chain_a2):等 tksel32_adapt 收官后,
#   接力 soupw1+稀有类加权 抬余量实验(hyb 适配已挪 B 并行)。
cd /workspace
echo "[chain_a3] 等待 tksel32_adapt 完成… $(date)"
while [ ! -f outputs/tksel32_adapt/eval_report.txt ]; do sleep 120; done
echo "[chain_a3] tksel32 完成,30s 后接力 soupw1_wt $(date)"
sleep 30
bash scripts/soupw1_wt.sh > outputs/soupw1_wt.log 2>&1
echo "[chain_a3] soupw1_wt 结束 $(date)"
