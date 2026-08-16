#!/bin/bash
# 廉价决定性探针:best-trained r64(soup_size/distill/train_params_best.npz, val2.2572@300, 仅无-dyn测过=79.68)
# 叠 dyn 推理(r64 上 dyn +0.44 SubKS 已验于 init 权重→80.12)。此"更好权重 + dyn"组合从未测过。
# 纯推理(~10min,不训练)。判据:vs 线 87.91/80.42;对照 init+dyn=87.56/80.12、distill无dyn=87.68/79.68。
cd /workspace && unset WDS_DIR
ts() { date '+%m-%d %H:%M'; }
O=outputs/r64distill_dyn
mkdir -p $O

echo "[r64distill-dyn $(ts)] 等 vfio 真空 …"
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

echo "[r64distill-dyn $(ts)] dyn 推理(8卡, best-trained distill 权重, --dump-letter-logits)"
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds outputs/soup_size/distill/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt

echo "[r64distill-dyn $(ts)] ==== best-trained r64 + dyn 汇总 ===="
echo "  校准: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: init+dyn=87.56/80.12 ; distill无dyn=87.68/79.68 ; 适配(dyn训练)=87.78/79.79 ; 线 87.91/80.42"
echo "  判据: 过线→int8复评定122MB交付(零代码改动); 否→合议重蒸(改代码)"
echo "[r64distill-dyn 完成 $(ts)]"
