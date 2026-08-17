#!/bin/bash
# 链式排满 · B机第四棒:等 hyb30 适配收官(含推理→eval_report)后,
#   自动接力 hyb2 升级臂(0812 用户确认"提高方法排到 hyb 后面")。
cd /workspace
echo "[chain_b4] 等待 tkhyb30_adapt 完成… $(date)"
while [ ! -f outputs/tkhyb30_adapt/eval_report.txt ]; do sleep 120; done
echo "[chain_b4] hyb30 完成,30s 后接力 hyb2 $(date)"
sleep 30
bash scripts/tkhyb2_adapt.sh > outputs/tkhyb2_adapt.log 2>&1
echo "[chain_b4] hyb2 结束 $(date)"
