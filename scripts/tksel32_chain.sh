#!/bin/bash
# Token 压缩实验链(用户 0812 "池化后选择,开始吧"):
#   ① 等 K=32 零样本评测(soupw1 原权重)收官 → 出裸分报告(空洞漂移代价)
#   ② 接适配重训:soupw1 暖启 + SELECT_TOKENS_K=32,700k 池 v1 增强,
#      与 soup 同口径隔离"选 token"单变量;训完带开关推理+报告。
# 判据:适配后公平校准 SubKS vs soupw1(81.17),差值=2×prefill 压缩净代价。
cd /workspace && unset WDS_DIR
Z=outputs/tksel32_zero
O=outputs/tksel32_adapt

echo "[chain] 等零样本评测收官… $(date)"
while [ ! -f $Z/eval_preds.jsonl ]; do sleep 60; done
python3 jax_impl/eval_metrics.py --preds $Z/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $Z/eval_report.txt
echo "[chain] 零样本报告已出,接适配重训 $(date)"

mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  SELECT_TOKENS_K=32 python jax_impl/train_sft.py $RESUME \
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
[ -f $O/train_params_best.npz ] || { echo "[tksel32] 无产物"; exit 1; }

SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[tksel32_chain] 完成 $(date)"
