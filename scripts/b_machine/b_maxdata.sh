#!/bin/bash
# 最大有用数据 anneal — B 机臂（用户 0811 05:20：加数据/两机并用/提准确率/暂不管压缩）。
#   B 机：从 seed2_base 全新 anneal 到 923k 全质量池（natural），prod rank，seed 11，
#   gentler lr（anneal_a 口径）。与 A 机 anneal_b_best→923k seed7 互为汤原料。
#   训练 + 基础评测；公平校准在 A 机集中做（把 eval_preds 回收）。
cd /workspace && unset WDS_DIR
O=outputs/maxdata_b
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_max_natural.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/seed2_base/seed2_best.npz \
  --augment --accum 32 \
  --lr 4e-6 --proj-lr 1e-4 --vision-lr 4e-6 \
  --warmup 100 --lr-schedule linear \
  --steps 2500 --eval-every 250 --early-stop-patience 6 \
  --ckpt-every 250 \
  --seed 11 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[maxdata_b] 无产物"; exit 1; }

INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[maxdata_b] 完成 $(date)"
