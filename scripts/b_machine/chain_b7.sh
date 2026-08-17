#!/bin/bash
# 通宵链 B 第七棒:tkhyb2b_s2 收官 → hyb2b 续退火(tkhyb2c)
cd /workspace
echo "[chain_b7] 等待 tkhyb2b_s2 完成… $(date)"
while [ ! -f outputs/tkhyb2b_s2/eval_report.txt ]; do sleep 120; done
echo "[chain_b7] s2 完成,接力 tkhyb2c 续退火 $(date)"
sleep 30
bash scripts/tkhyb2c.sh > outputs/tkhyb2c.log 2>&1
echo "[chain_b7] 结束 $(date)"
