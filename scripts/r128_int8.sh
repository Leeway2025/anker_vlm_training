#!/bin/bash
# r128 阶梯的 int8 复评(约240MB):r128 无蒸馏截断 fp32=80.63 过线,测 int8 后是否守线。
# 用 select K=32(r128 源自 soup_tk32),uniform-128 → auto 判秩(勿加 --rank-scheme prod)。
set -e
cd /workspace && unset WDS_DIR
export SELECT_TOKENS_K=32
O=outputs/soup_tk32_r128
DEL=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }
[ -f $O/model.npz ] || { echo "[$(ts)] 无 r128 model.npz,退出"; exit 1; }

echo "[$(ts)] r128 int8 逐通道量化"
python3 $DEL/quantize_lora.py --in $O/model.npz \
  --out-int8real $O/model_int8.npz --out-bf16 $O/model_bf16.npz 2>&1 | tail -3
python3 $DEL/dequant_int8.py --in $O/model_int8.npz --out $O/model_int8_dequant.npz
echo "[$(ts)] r128 int8 反量化评测(K=32,8卡,auto 判秩)"
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/qeval_int8 $O/model_int8_dequant.npz 8
python3 jax_impl/eval_metrics.py --preds $O/qeval_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/qeval_int8_report.txt
echo "[$(ts)] === r128 int8 公平校准(class_diag)==="
python3 outputs/class_diag.py $O/qeval_int8.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib_int8.txt
echo "[$(ts)] ===== r128-int8 汇总(线 87.91/80.42)====="
echo "  r128 fp32: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  r128 int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
ls -la $O/model_int8.npz $O/model_bf16.npz 2>/dev/null
echo "[$(ts)] r128_int8 完成"
