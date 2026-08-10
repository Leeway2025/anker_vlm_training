#!/bin/bash
# B机夜班第二棒: 等 night_soup 完 → git pull(取变秩支持与修补脚本)
# → 等 A机把 trunc_r64/model.npz + teacher_u512.npz 推过来
# → r64 蒸馏修补(400步)+ 评测 → 报告回传 GCS 由宿主机守望进程负责。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }

while ! grep -q 'B机夜班完成' logs/night_soup.log 2>/dev/null; do sleep 300; done
sleep 30
git pull --ff-only 2>&1 | tail -1

while [ ! -f $M/.staged_ok ]; do
  echo "[$(ts)] 等 A机 staging(trunc_r64 + teacher_u512)"; sleep 300
done
bash scripts/repair_distill.sh 64 400
echo "[$(ts)] B机修补棒完成"
