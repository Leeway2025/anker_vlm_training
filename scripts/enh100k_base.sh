#!/bin/bash
# 100k 增强试验 · A机=基线臂(对照,无加权)。用户 0811 07:20:增强先100k试再700k。
#   从 anneal_b_best 续 600 步到共享 100k(labels_100k_v2),prod rank,seed 7。
#   与 B机 enh100k_wt(同数据同超参,仅多 --sample-weights)配对,隔离"稀有类加权"效应。
#   只训练+推理+裸评测;校准在 A 机集中对两臂统一做(消除校准代码漂移)。
cd /workspace && unset WDS_DIR
O=outputs/enh100k_base
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_100k_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/anneal_b_best.npz \
  --augment --accum 32 \
  --lr 8e-6 --proj-lr 2e-4 --vision-lr 8e-6 \
  --warmup 40 --lr-schedule linear \
  --steps 600 --eval-every 100 --early-stop-patience 4 \
  --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[enh100k_base] 无产物"; exit 1; }

INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[enh100k_base] 完成 $(date)"
