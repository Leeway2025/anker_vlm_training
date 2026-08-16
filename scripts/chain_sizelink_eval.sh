#!/bin/bash
# 等 TPU 空(soup_tk32_r128 infer 释放 vfio)后,跑体积链 r64 蒸馏的评测尾段。
cd /workspace
echo "[chain-sizelink] 等 vfio 释放 …"
while :; do
  hold=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    comm=$(cat /proc/$p/comm 2>/dev/null)
    case "$comm" in python|python3) : ;; *) continue;; esac
    if ls -l /proc/$p/fd 2>/dev/null | grep -q vfio; then hold=1; break; fi
  done
  [ "$hold" = 0 ] && break
  sleep 20
done
echo "[chain-sizelink] vfio 空,启动 sizelink_eval $(date)"
bash scripts/sizelink_eval.sh 2>&1 | tee -a outputs/soup_size/sizelink_eval.log
