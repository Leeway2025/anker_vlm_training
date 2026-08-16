#!/bin/bash
# 链式排满 · A机:等增强基线臂 enh_aug_base 完全跑完(含推理→eval_report)后,
#   立即接力 700k 主力大跑,零空档。用户 0811 "尽量排满"。
cd /workspace
echo "[chain_a] 等待 enh_aug_base 完成… $(date)"
while [ ! -f outputs/enh_aug_base/eval_report.txt ]; do sleep 120; done
echo "[chain_a] enh_aug_base 完成,30s 后接力 700k $(date)"
sleep 30
bash scripts/run_700k_enh.sh > outputs/run_700k_enh.log 2>&1
echo "[chain_a] 700k 结束 $(date)"
