#!/bin/bash
# 等 vfio 释放(soup_tk32_r96 infer 收尾)后,跑 8帧x64 零样本探针。
cd /workspace
echo "[chain-probe8f] 等 vfio 释放 …"
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
echo "[chain-probe8f] vfio 空,启动 probe_8f64 $(date)"
bash scripts/probe_8f64.sh 2>&1 | tee -a outputs/probe_8f64.log
