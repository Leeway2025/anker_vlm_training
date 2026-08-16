#!/bin/bash
# 仅跑 cb_moderate 的评测尾巴(训练已完成,复用 train_params_best.npz)
set -e
cd /workspace && unset WDS_DIR
O=outputs/cb_moderate_r64
ts() { date '+%m-%d %H:%M'; }
[ -s $O/train_params_best.npz ] || { echo "缺 best"; exit 1; }

echo "[tail $(ts)] 等 vfio 真空 …"
for i in $(seq 1 30); do
  hold=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    comm=$(cat /proc/$p/comm 2>/dev/null) || true
    case "$comm" in python|python3) : ;; *) continue;; esac
    if ls -l /proc/$p/fd 2>/dev/null | grep -q vfio; then hold=1; break; fi
  done
  [ "$hold" = 0 ] && break
  sleep 10
done

echo "[tail $(ts)] fp32 dyn 推理"
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl --labels /data/labels_test.jsonl | tee $O/eval_report.txt

echo "[tail $(ts)] int8 量化 + 推理"
python3 outputs/delivery_0807/quantize_lora.py --in $O/train_params_best.npz \
  --out-bf16 $O/model_bf16.npz --out-int8 $O/model_int8sim.npz
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds_int8 $O/model_int8sim.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds_int8.jsonl --labels /data/labels_test.jsonl | tee $O/eval_report_int8.txt

echo "[tail $(ts)] class_diag 校准"
calib() {
  [ -s "$1" ] || { echo "缺 $1"; return 1; }
  python3 outputs/class_diag.py "$1" --gold /data/labels_test.jsonl \
    --train /data/labels_train_plus_testval_v2.jsonl 2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}
calib $O/eval_preds.jsonl      $O/fair_calib_fp32.txt "fp32"
calib $O/eval_preds_int8.jsonl $O/fair_calib_int8.txt "int8"
echo "[tail $(ts)] ==== cb_moderate 汇总 ===="
echo "  fp32: $(cat $O/fair_calib_fp32.txt 2>/dev/null | tr '\n' ' ')"
echo "  int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: 线 87.91/80.42 ; 冠军 int8=87.78/80.16 ; cb_aggr int8=87.69/79.70"
echo "[tail 完成 $(ts)]"
