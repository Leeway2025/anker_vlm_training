#!/bin/bash
# r128 int4 全量评测:r128 fp32=80.63 过线,int4=121.7MB 卡进 122MB 预算,测守不守线。
# 用 select K=32(r128 源自 soup_tk32),uniform-128 → auto 判秩(勿加 --rank-scheme prod)。
set -e
cd /workspace && unset WDS_DIR
export SELECT_TOKENS_K=32
O=outputs/soup_tk32_r128
ts() { date '+%m-%d %H:%M'; }
[ -f $O/model_int4_dequant.npz ] || { echo "[$(ts)] 无 int4_dequant,退出"; exit 1; }

echo "[$(ts)] r128 int4 反量化评测(K=32,8卡,auto 判秩)"
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/qeval_int4 $O/model_int4_dequant.npz 8
python3 jax_impl/eval_metrics.py --preds $O/qeval_int4.jsonl \
  --labels /data/labels_test.jsonl | tee $O/qeval_int4_report.txt
echo "[$(ts)] === r128 int4 公平校准(class_diag)==="
python3 outputs/class_diag.py $O/qeval_int4.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib_int4.txt
echo "[$(ts)] ===== r128-int4 汇总(线 87.91/80.42,预算 122MB)====="
echo "  r128 fp32: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  r128 int4: $(cat $O/fair_calib_int4.txt 2>/dev/null | tr '\n' ' ')  size=121.7MB"
echo "[$(ts)] r128_int4 完成"
