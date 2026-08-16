#!/bin/bash
# r96_recover 交付判据:重推理带 --dump-letter-logits → 公平校准(class_diag)出 SK_cal vs 线 80.42。
# 原 chain 的 infer 没带 logits,无法校准;这里补齐。K=32、uniform r96、layout=16帧 hf_layout.json。
set -e
cd /workspace && unset WDS_DIR
O=outputs/r96_recover
ts() { date '+%m-%d %H:%M'; }
echo "[r96-calib $(ts)] 8卡重推理带字母logits(K=32, r96, 16帧)"
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds_ll $O/train_params_best.npz 8
echo "[r96-calib $(ts)] 官方裸评测"
python3 jax_impl/eval_metrics.py --preds $O/eval_preds_ll.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report_ll.txt
echo "[r96-calib $(ts)] 公平校准 class_diag"
python3 outputs/class_diag.py $O/eval_preds_ll.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt
echo "[r96-calib $(ts)] ==== r96_recover 汇总 ===="
echo "  校准: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')  (线 RT87.91/SubKS80.42; 未恢复r96校准=87.85/80.37)"
echo "[r96-calib] 完成 $(ts)"
