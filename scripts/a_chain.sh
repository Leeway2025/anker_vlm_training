#!/bin/bash
# 串联:变秩恢复(soup+非soup)跑完 → 再跑非汤 uniform r64 + 汤蒸馏,凑齐三候选。
# 注意:容器内无 pgrep,用日志完成哨兵判断,而非进程探测。
cd /workspace && unset WDS_DIR
LOG=outputs/delivery_0807/varrank_recover.log
ts(){ date '+%m-%d %H:%M'; }
echo "[$(ts)] chain: 等 a_varrank_recover 完成哨兵(日志 'A机核心实验完成')"
until grep -q 'A机核心实验完成' "$LOG" 2>/dev/null; do sleep 120; done
echo "[$(ts)] chain: 变秩恢复已完成,启动 a_nonsoup(uniform r64 + 蒸馏)"
bash scripts/a_nonsoup.sh
echo "[$(ts)] chain: 全部完成"
