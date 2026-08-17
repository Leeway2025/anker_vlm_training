#!/bin/bash
# B机压缩第二棒(0812 改排:用户"K=48 无意义,核心=适应飘忽空洞"→
#   撤 K=48,hyb30 适配挪到 B 并行,方法终判提前;文件名保留供 chain_b3 调用)。
#   ① hyb30 零样本(soupw1 原权重,~25min):与纯选择零样本 77.24 同预算比。
#   ② hyb30 适配重训:与 A 机 tksel32_adapt 同配方,唯一差异=压缩方式。
cd /workspace && unset WDS_DIR

Z=outputs/tkhyb30_zero
if [ ! -f $Z/eval_report.txt ]; then
  mkdir -p $Z
  TOKEN_COMPRESS_MODE=hyb SELECT_TOKENS_K=30 \
    INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
    bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json \
    $Z/eval_preds outputs/soupw1/soupw1.npz 8 && \
  python3 jax_impl/eval_metrics.py --preds $Z/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $Z/eval_report.txt
fi

O=outputs/tkhyb30_adapt
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  TOKEN_COMPRESS_MODE=hyb SELECT_TOKENS_K=30 python jax_impl/train_sft.py $RESUME \
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
[ -f $O/train_params_best.npz ] || { echo "[tkhyb30] 无产物"; exit 1; }

TOKEN_COMPRESS_MODE=hyb SELECT_TOKENS_K=30 \
  INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[tkhyb30_adapt@B] 完成 $(date)"
