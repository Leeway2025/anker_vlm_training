#!/bin/bash
# 等 vfio 释放(tkdyn32_zero 等在跑的探针收尾)后,补跑 r96_recover 公平校准。
# 交付判据:r96_recover(180MB档短恢复)校准 SubKS 是否过 80.42。
cd /workspace
ts() { date '+%m-%d %H:%M'; }
echo "[chain-r96-calib $(ts)] 等 vfio 释放 …"
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
echo "[chain-r96-calib $(ts)] vfio 空,清理失败残片 → 起校准"
rm -f outputs/r96_recover/eval_preds_ll_shard* outputs/r96_recover/eval_preds_ll.jsonl 2>/dev/null
sleep 3
bash scripts/r96_recover_calib.sh
