#!/bin/bash
# 链式排满 · B机第二棒:等 v3wt 完全跑完(含推理→eval_report)后,
#   立即接力 700k 分歧成员 run_700k_v1(v1-only+加权/seed2),零空档。用户 0811 "尽量排满/两机并用/提准确率"。
cd /workspace
echo "[chain_b2] 等待 enh_aug_v3wt 完成… $(date)"
while [ ! -f outputs/enh_aug_v3wt/eval_report.txt ]; do sleep 120; done
echo "[chain_b2] v3wt 完成,30s 后接力 700k_v1 $(date)"
sleep 30
bash scripts/run_700k_v1.sh > outputs/run_700k_v1.log 2>&1
echo "[chain_b2] 700k_v1 结束 $(date)"
