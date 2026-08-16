#!/bin/bash
# 客户 0815 线索:对方"只训前3字符标签(RT|分隔符|SubKS)"= 把 caption 从损失里砍掉,
# 梯度全压到打分位。我们诊断早有教训#10(val_loss 被 ~60 caption token 主导、打分两位
# 几乎不进梯度)。本实验单变量验证这条:与 tksel32_adapt 完全同配(soupw1 暖启 + K=32),
# 唯一差别 CAP_WEIGHT=0(caption 段权重归零)。
# 判据:公平校准 SubKS vs 基线 tksel32_adapt(同配 CAP_WEIGHT=1)。
cd /workspace && unset WDS_DIR
O=outputs/capmask_tk32
B=outputs/tksel32_adapt        # 同配基线(CAP_WEIGHT=1),用于校准对比
mkdir -p $O

echo "[capmask] 训练开始 $(date) —— CAP_WEIGHT=0, 只训 RT|SubKS 打分位"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  CAP_WEIGHT=0 SELECT_TOKENS_K=32 python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/soupw1/soupw1.npz \
  --augment --accum 32 \
  --steps 1500 --eval-every 100 --early-stop-patience 3 \
  --ckpt-every 400 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[capmask] 无产物"; exit 1; }

echo "[capmask] 推理 K=32 $(date)"
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt

echo "[capmask] === 公平校准(class_diag, n=11022)==="
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl \
  --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt

echo "[capmask] === 基线 tksel32_adapt 同口径校准(对照)==="
if [ -f $B/eval_preds.jsonl ]; then
  python3 outputs/class_diag.py $B/eval_preds.jsonl \
    --gold /data/labels_test.jsonl \
    --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $B/fair_calib.txt
fi

echo "[capmask] ===== 汇总(线 87.91/80.42)====="
echo "  基线 tksel32_adapt (CAP_WEIGHT=1): $(cat $B/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  capmask       (CAP_WEIGHT=0): $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "[capmask_tk32] 完成 $(date)"
