#!/bin/bash
# 链式排满 · B机第三棒:等 run_700k_v1 完全跑完(含推理→eval_report)后,
#   立即接力 token 压缩 K=48 适配臂。用户 0812 "不能低于指标线"→K轴第二点。
cd /workspace
echo "[chain_b3] 等待 run_700k_v1 完成… $(date)"
while [ ! -f outputs/run_700k_v1/eval_report.txt ]; do sleep 120; done
echo "[chain_b3] 700k_v1 完成,30s 后接力 tksel48 $(date)"
sleep 30
bash scripts/tksel48_adapt.sh > outputs/tksel48_adapt.log 2>&1
echo "[chain_b3] tksel48 结束 $(date)"
