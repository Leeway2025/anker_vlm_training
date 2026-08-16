#!/bin/bash
# 链式排满 · A机第二棒 v2(0812 策略review后替代 chain_a3):
#   tksel32_adapt 收官 → ①MST=32 原生降分辨率零样本探针(~25min,
#   分布内压缩、双省编码器+prefill,NPU零改动的候选)→ ②soupw1_wt 加权抬余量。
cd /workspace
echo "[chain_a4] 等待 tksel32_adapt 完成… $(date)"
while [ ! -f outputs/tksel32_adapt/eval_report.txt ]; do sleep 120; done
echo "[chain_a4] tksel32 完成,30s 后跑 MST=32 零样本探针 $(date)"
sleep 30

Z=outputs/tkmst32_zero
if [ ! -f $Z/eval_report.txt ]; then
  mkdir -p $Z
  MAX_SOFT_TOKENS=32 \
    INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
    bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json \
    $Z/eval_preds outputs/soupw1/soupw1.npz 8 && \
  python3 jax_impl/eval_metrics.py --preds $Z/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $Z/eval_report.txt
fi

echo "[chain_a4] 探针完成,接力 soupw1_wt $(date)"
bash scripts/soupw1_wt.sh > outputs/soupw1_wt.log 2>&1
echo "[chain_a4] soupw1_wt 结束 $(date)"
