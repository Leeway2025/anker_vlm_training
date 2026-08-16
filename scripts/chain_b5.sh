#!/bin/bash
# 链式排满 · B机重排(0812 用户"合并明显不如纯选"→hyb2 提前,hyb30 降级):
#   ① hyb2 零样本+适配(scripts/tkhyb2_adapt.sh,最可能的交付臂,优先)
#   ② 之后接 hyb30 适配(机制验证臂,可选;其零样本已出 77.06,
#      tksel48_adapt.sh 的零样本段会因 report 已存在自动跳过)
cd /workspace
echo "[chain_b5] 起跑:hyb2 优先 $(date)"
bash scripts/tkhyb2_adapt.sh > outputs/tkhyb2_adapt.log 2>&1
echo "[chain_b5] hyb2 完成,接力 hyb30 适配(机制验证)$(date)"
bash scripts/tksel48_adapt.sh > outputs/tkhyb30_retry.log 2>&1
echo "[chain_b5] 全部结束 $(date)"
