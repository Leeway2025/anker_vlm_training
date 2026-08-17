#!/bin/bash
# 链式排满 · B机:等增强臂 enh_aug_v3 完全跑完(含推理→eval_report)后,
#   立即接力 增强+加权叠加确认臂 enh_aug_v3wt,零空档。用户 0811 "尽量排满"。
cd /workspace
echo "[chain_b] 等待 enh_aug_v3 完成… $(date)"
while [ ! -f outputs/enh_aug_v3/eval_report.txt ]; do sleep 120; done
echo "[chain_b] enh_aug_v3 完成,30s 后接力 v3wt $(date)"
sleep 30
bash scripts/enh_aug_v3wt.sh > outputs/enh_aug_v3wt.log 2>&1
echo "[chain_b] v3wt 结束 $(date)"
