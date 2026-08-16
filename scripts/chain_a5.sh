#!/bin/bash
# 通宵自动链 A(0812 用户"跑完自行往目标卷"):tksel32b 收官→seed2 成员接力
cd /workspace
echo "[chain_a5] 等待 tksel32b 完成… $(date)"
while [ ! -f outputs/tksel32b/eval_report.txt ]; do sleep 120; done
echo "[chain_a5] tksel32b 完成,接力 seed2 成员 $(date)"
sleep 30
bash scripts/tksel32b_s2.sh > outputs/tksel32b_s2.log 2>&1
echo "[chain_a5] 结束 $(date)"
