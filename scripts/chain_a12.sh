#!/bin/bash
# A机主链(0814晚):等learnhead释放 → ①受害集K48反事实(3min)
#   → ②60k训练集错误标注(r64v3门槛,~3h)→ ③显著性提取(~4h)
cd /workspace
while ls /proc/*/cmdline 2>/dev/null | xargs -I{} sh -c "tr \"\0\" \" \" < {} 2>/dev/null" 2>/dev/null | grep -v chain_a12 | grep -qE "train_sft|infer\.py"; do sleep 180; done
sleep 90
# ① 受害集 K=48 反事实
mkdir -p outputs/victims_k48
SELECT_TOKENS_K=48 INFER_ARGS="--rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python outputs/victims_labels.jsonl /data/hf_layout.json \
  outputs/victims_k48/eval_preds outputs/soup_tk32/train_params_best.npz 8 > outputs/victims_k48.log 2>&1
python3 jax_impl/eval_metrics.py --preds outputs/victims_k48/eval_preds.jsonl \
  --labels outputs/victims_labels.jsonl > outputs/victims_k48/eval_report.txt 2>&1
# ② 60k 训练子集错误标注(学生=r64合议KD best,K=32 uniform 口径)
mkdir -p outputs/train60k_pred
SELECT_TOKENS_K=32 INFER_ARGS="" \
  bash jax_impl/infer_sharded.sh python outputs/train60k.jsonl /data/hf_layout.json \
  outputs/train60k_pred/eval_preds outputs/r64_ens_kd/train_params_best.npz 8 > outputs/train60k_pred.log 2>&1
# ③ 显著性提取(8分片)
for i in 0 1 2 3 4 5 6 7; do
  TPU_VISIBLE_CHIPS=$i TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_ADDRESSES=localhost:$((8576+i)) TPU_PROCESS_PORT=$((8576+i)) CLOUD_TPU_TASK_ID=0 \
  python3 jax_impl/extract_saliency.py --init-npz outputs/soupw1/soupw1.npz \
    --labels /data/labels_train_plus_testval_v2.jsonl --layout /data/hf_layout.json \
    --n 50000 --shard $i/8 --out outputs/saliency/sal_shard$i.npz > outputs/saliency/shard$i.log 2>&1 &
done
wait
echo "[chain_a12] 全部完成 $(date)" > outputs/chain_a12.DONE
