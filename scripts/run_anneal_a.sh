#!/bin/bash
# A机退火续训(温和档): 从 seed2 best 起, 短schedule衰到0, test-val v2 选卡
# peak lr=4e-6(prod 0.2x) steps=1600 (~7.6h) — 先出结果当早期信号
# 地板=seed2 73.69, 任何时候不如它就交 seed2
cd /workspace && unset WDS_DIR
ATT=0
until RESUME=""; [ -f outputs/jax_anneal_a/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/seed2_base/seed2_best.npz \
  --augment --accum 32 \
  --lr 4e-6 --proj-lr 1e-4 --vision-lr 4e-6 \
  --warmup 100 --lr-schedule linear \
  --steps 1600 --eval-every 200 --early-stop-patience 5 \
  --ckpt-every 250 --resume \
  --seed 4 --mu-dtype float32 --prefetch-workers 24 \
  --out outputs/jax_anneal_a; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 8 ] && exit 1
  sleep 60
done
INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  outputs/jax_anneal_a/eval_preds outputs/jax_anneal_a/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds outputs/jax_anneal_a/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee outputs/jax_anneal_a/eval_report.txt
