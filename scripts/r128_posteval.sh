#!/bin/bash
# r128 无蒸馏截断体积链的评测尾(CPU,无需 vfio):等 infer 合并出 eval_preds.jsonl → 指标+公平校准。
cd /workspace
O=outputs/soup_tk32_r128
ts() { date '+%m-%d %H:%M'; }
echo "[r128-post] 等 $O/eval_preds.jsonl 合并完成 …"
# 等到文件存在且 8 卡 infer 进程退出(避免读到半成品)
while :; do
  [ -f $O/eval_preds.jsonl ] || { sleep 30; continue; }
  running=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    grep -qa "soup_tk32_r128" /proc/$p/cmdline 2>/dev/null && grep -qa "infer.py" /proc/$p/cmdline 2>/dev/null && { running=1; break; }
  done
  [ "$running" = 0 ] && break
  sleep 30
done
echo "[r128-post] preds 就绪,评测 $(date)"
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[r128-post] === r128 公平校准(class_diag)==="
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt
echo "[r128-post] r128 汇总: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')  size=$(ls -la $O/model.npz|awk '{print $5}')"
echo "[r128-post] 完成 $(ts)"
