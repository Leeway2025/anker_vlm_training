#!/bin/bash
# 体积链·续跑尾段(0813):蒸馏已完成(train_params_best.npz best_val 2.2572@300 已存)。
# 只跑 ③fp32评测+校准 与 ④int8量化+反量化评测+校准。
# 关键修正:uniform-r64 学生 → 推理用 auto 判秩(勿加 --rank-scheme prod,否则 506 叶不命中)。
set -e
cd /workspace && unset WDS_DIR
export SELECT_TOKENS_K=32
O=outputs/soup_size
DEL=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }
[ -f $O/distill/train_params_best.npz ] || { echo "[$(ts)] 无蒸馏产物,退出"; exit 1; }

echo "[$(ts)] ③ fp32 评测(K=32,8卡,auto 判秩)"
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/distill/eval_preds $O/distill/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/distill/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/distill/eval_report.txt
echo "[$(ts)] === fp32 公平校准(class_diag)==="
python3 outputs/class_diag.py $O/distill/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/distill/fair_calib_fp32.txt

echo "[$(ts)] ④ int8 逐通道量化 + 反量化评测"
python3 $DEL/quantize_lora.py --in $O/distill/train_params_best.npz \
  --out-int8real $O/model_int8.npz --out-bf16 $O/model_bf16.npz 2>&1 | tail -3
python3 $DEL/dequant_int8.py --in $O/model_int8.npz --out $O/model_int8_dequant.npz
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/qeval_int8 $O/model_int8_dequant.npz 8
python3 jax_impl/eval_metrics.py --preds $O/qeval_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/qeval_int8_report.txt
echo "[$(ts)] === int8 公平校准(class_diag)==="
python3 outputs/class_diag.py $O/qeval_int8.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib_int8.txt

echo "[$(ts)] ===== 体积链汇总(公平校准口径,线 87.91/80.42)====="
echo "  fp32-r64: $(cat $O/distill/fair_calib_fp32.txt 2>/dev/null | tr '\n' ' ')"
echo "  int8-r64: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
ls -la $O/model_int8.npz $O/model_bf16.npz 2>/dev/null
echo "[$(ts)] sizelink_eval 完成"
