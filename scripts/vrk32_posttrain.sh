#!/bin/bash
# 后处理(CPU-only,不占卡):兄弟链 vrk32_distill.sh 已在 K32 下产出带 letter-logits 的
# eval_preds.jsonl(fp32)与 eval_preds_int8.jsonl(int8),但只跑了 eval_metrics 裸评、
# 缺 class_diag 公平校准(与线 87.91/80.42 同口径)。本脚本等两份 preds 落地后,对二者
# 各跑一遍 class_diag → fair_calib_fp32.txt / fair_calib_int8.txt。纯 CPU,无 vfio 争用。
cd /workspace && unset WDS_DIR
ts() { date '+%m-%d %H:%M'; }
O=outputs/vrk32_distill

calib() {  # $1=preds $2=out.txt $3=标签
  [ -s "$1" ] || { echo "[vrk32-calib $(ts)] 缺 $1"; return 1; }
  echo "[vrk32-calib $(ts)] class_diag $3 ($(wc -l < $1) 行)"
  python3 outputs/class_diag.py "$1" \
    --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}

echo "[vrk32-calib $(ts)] 等兄弟链完工标记(或 int8 preds 稳定)…"
N=0
while [ $N -lt 240 ]; do
  if grep -q "vrk32_distill] 完成" outputs/vrk32_distill.log 2>/dev/null; then break; fi
  N=$((N+1)); sleep 30
done

echo "[vrk32-calib $(ts)] 兄弟链完工;跑公平校准(CPU)"
calib $O/eval_preds.jsonl      $O/fair_calib_fp32.txt "fp32"
calib $O/eval_preds_int8.jsonl $O/fair_calib_int8.txt "int8"

echo "[vrk32-calib $(ts)] ==== vr×K32(u512蒸馏) 公平校准汇总 ===="
echo "  fp32: $(cat $O/fair_calib_fp32.txt 2>/dev/null | tr '\n' ' ')"
echo "  int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: vr全token=88.00/80.46 ; vr+K32零样本=84.66/76.01 ; r64+dyn=87.56/80.12 ; 线 87.91/80.42"
echo "  判据: int8 过线→u512 路径就够(省合议); 否→上合议双老师重蒸(基建已就位)"
echo "[vrk32-calib 完成 $(ts)]"
