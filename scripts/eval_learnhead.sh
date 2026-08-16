#!/bin/bash
# ①学习打分头 · 硬验收:把 head-only 训出的 tok_scorer(step100, val_loss2.2553)
# 配回底座 = learnhead_r64/train_params_best.npz(base 与 ens_r64_dyn 逐字节同,lr=0
# 冻结,只多了 tok_scorer_A/b)。dyn K=32 训推一致;infer 侧 TOKEN_LEARN_SCORE=1
# TOKEN_LEARN_GAIN=8(与训练同增益)、不设 TOKEN_LEARN_TRAIN(纯硬 top_k,端侧可导出)。
# fp32 → int8(tok_scorer 保 fp,量化 base)→ class_diag 公平校准(同线口径)。
# 判据:int8 SubKS 过 80.42 → ① 成立定档;flat/负 → 学习打分头杠杆耗尽,如实上报。
set -e
cd /workspace && unset WDS_DIR
ts() { date '+%m-%d %H:%M'; }
P=outputs/learnhead_r64/train_params_best.npz
O=outputs/learnhead_r64_eval
mkdir -p $O
[ -s "$P" ] || { echo "[eval-lh $(ts)] 缺头产物 $P"; exit 1; }

export TOKEN_LEARN_SCORE=1 TOKEN_LEARN_GAIN=8
export SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn
# 注意:不 export TOKEN_LEARN_TRAIN(=推理纯硬 top_k)

echo "[eval-lh $(ts)] 等 vfio 真空 …"
while :; do
  hold=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    comm=$(cat /proc/$p/comm 2>/dev/null) || true
    case "$comm" in python|python3) : ;; *) continue;; esac
    if ls -l /proc/$p/fd 2>/dev/null | grep -q vfio; then hold=1; break; fi
  done
  [ "$hold" = 0 ] && break
  sleep 20
done

echo "[eval-lh $(ts)] fp32 dyn 推理(带头, GAIN=8, 硬 top_k)"
INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $P 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt

echo "[eval-lh $(ts)] int8 量化(tok_scorer 保 fp)"
python3 outputs/delivery_0807/quantize_lora.py --in $P \
  --out-bf16 $O/model_bf16.npz --out-int8 $O/model_int8sim.npz
INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds_int8 $O/model_int8sim.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report_int8.txt

calib() {  # $1=preds $2=out.txt $3=tag
  [ -s "$1" ] || { echo "[eval-lh $(ts)] 缺 $1"; return 1; }
  echo "[eval-lh $(ts)] class_diag $3 ($(wc -l < $1) 行)"
  python3 outputs/class_diag.py "$1" \
    --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}
calib $O/eval_preds.jsonl      $O/fair_calib_fp32.txt "fp32"
calib $O/eval_preds_int8.jsonl $O/fair_calib_int8.txt "int8"

echo "[eval-lh $(ts)] ==== ①学习打分头 硬验收汇总 ===="
echo "  fp32: $(cat $O/fair_calib_fp32.txt 2>/dev/null | tr '\n' ' ')"
echo "  int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: 线 87.91/80.42 ; 底座 ens_r64_dyn int8=87.78/80.16(缺 0.26)"
echo "  int8 大小: $(du -h $O/model_int8sim.npz 2>/dev/null | cut -f1)"
echo "[eval-lh 完成 $(ts)]"
