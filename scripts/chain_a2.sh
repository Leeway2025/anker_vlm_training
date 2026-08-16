#!/bin/bash
# 链式排满 · A机第二棒:等 tksel32_adapt 完全收官(含推理→eval_report)后,
#   立即接力 hyb(2+3+4合一)适配臂。用户 0812 "只测选择不够,2 3 4 一起"。
cd /workspace
echo "[chain_a2] 等待 tksel32_adapt 完成… $(date)"
while [ ! -f outputs/tksel32_adapt/eval_report.txt ]; do sleep 120; done
echo "[chain_a2] tksel32 完成,30s 后接力 tkhyb30 $(date)"
sleep 30
bash scripts/tkhyb30_adapt.sh > outputs/tkhyb30_adapt.log 2>&1
echo "[chain_a2] tkhyb30 结束 $(date)"
