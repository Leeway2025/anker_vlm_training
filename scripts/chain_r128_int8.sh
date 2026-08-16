#!/bin/bash
# 等 r64-int8 评测出结果 + vfio 释放,再跑 r128-int8 复评(约240MB 中等体积候选)。
cd /workspace
echo "[chain-r128int8] 等 r64-int8 校准出 + vfio 释放 …"
while [ ! -f outputs/soup_size/fair_calib_int8.txt ]; do sleep 30; done
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
echo "[chain-r128int8] vfio 空,启动 r128_int8 $(date)"
bash scripts/r128_int8.sh 2>&1 | tee -a outputs/soup_tk32_r128/r128_int8.log
