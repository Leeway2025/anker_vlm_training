#!/bin/bash
# 通宵自动链 B(0812 用户"跑完自行往目标卷"):tkhyb2b 收官→seed11 成员接力
cd /workspace
echo "[chain_b6] 等待 tkhyb2b 完成… $(date)"
while [ ! -f outputs/tkhyb2b/eval_report.txt ]; do sleep 120; done
echo "[chain_b6] tkhyb2b 完成,接力 seed11 成员 $(date)"
sleep 30
bash scripts/tkhyb2b_s2.sh > outputs/tkhyb2b_s2.log 2>&1
echo "[chain_b6] 结束 $(date)"
