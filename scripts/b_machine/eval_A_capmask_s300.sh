#!/bin/bash
# Parallel eval on B of A's capmask_r64_dyn step300 (LAST ckpt, train_params.npz).
# Tests the "wrong checkpoint" hypothesis: early-stop picked best@100 on caption-free
# val_loss; does SubKS peak later? Mirrors eval_A_capmask.sh. Output: capmask_A_eval_s300.
set -e
cd /workspace && unset WDS_DIR
O=outputs/capmask_A_eval_s300
ts() { date '+%m-%d %H:%M'; }
echo "[Beval-s300 $(ts)] start on $(hostname); ckpt:"
ls -la $O/train_params.npz
[ -s $O/train_params.npz ] || { echo "[Beval-s300] no ckpt"; exit 1; }

echo "[Beval-s300 $(ts)] fp32 dyn infer (K=32)"
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt

echo "[Beval-s300 $(ts)] int8 quantize + dyn infer"
python3 outputs/delivery_0807/quantize_lora.py --in $O/train_params.npz \
  --out-bf16 $O/model_bf16.npz --out-int8 $O/model_int8sim.npz
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds_int8 $O/model_int8sim.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report_int8.txt

calib() {
  [ -s "$1" ] || { echo "[Beval-calib $(ts)] missing $1"; return 1; }
  echo "[Beval-calib $(ts)] class_diag $3 ($(wc -l < $1) rows)"
  python3 outputs/class_diag.py "$1" \
    --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}
calib $O/eval_preds.jsonl      $O/fair_calib_fp32.txt "fp32"
calib $O/eval_preds_int8.jsonl $O/fair_calib_int8.txt "int8"

echo "[Beval-s300 $(ts)] ==== A step300 capmask (CAP_WEIGHT=0) on B ===="
echo "  fp32: $(cat $O/fair_calib_fp32.txt 2>/dev/null | tr '\n' ' ')"
echo "  int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  line 87.91/80.42 ; champ int8=80.16 ; step100=79.98"
echo "[Beval-s300 done $(ts)]"
