#!/bin/bash
# r96 无蒸馏截断的评测尾(CPU,无需 vfio):等 infer 合并 eval_preds.jsonl → 指标+公平校准。
# 补全秩曲线数据点(r64=79.68 / r96=? / r128=80.63);客户要≤122MB,r96仅参考。
cd /workspace
O=outputs/soup_tk32_r96
ts() { date '+%m-%d %H:%M'; }
echo "[r96-post] 等 $O/eval_preds.jsonl 合并 + infer 退出 …"
while :; do
  [ -f $O/eval_preds.jsonl ] || { sleep 30; continue; }
  running=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    grep -qa "soup_tk32_r96" /proc/$p/cmdline 2>/dev/null && grep -qa "infer.py" /proc/$p/cmdline 2>/dev/null && { running=1; break; }
  done
  [ "$running" = 0 ] && break
  sleep 30
done
echo "[r96-post] preds 就绪,评测 $(date)"
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt
echo "[r96-post] r96 汇总: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')  size=$(ls -la $O/model.npz 2>/dev/null|awk '{print $5}')"
echo "[r96-post] 完成 $(ts)"
