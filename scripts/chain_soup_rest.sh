#!/bin/bash
# 等 soup_tk32 报告出现 + vfio 真正释放,再跑剩余汤口径。避免 teardown 滞后撞 vfio。
cd "$(dirname "$0")/.."
echo "[chain-soup-rest] 等 soup_tk32/eval_report.txt …"
while [ ! -f outputs/soup_tk32/eval_report.txt ]; do sleep 60; done
echo "[chain-soup-rest] 报告已出,等 8 卡 infer 释放 vfio …"
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
echo "[chain-soup-rest] vfio 空,启动 soup_k32_rest $(date)"
bash scripts/soup_k32_rest.sh
