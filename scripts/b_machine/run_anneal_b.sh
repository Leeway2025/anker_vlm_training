#!/bin/bash
# B机退火续训(深挖档): 从 seed2 best 起, 短schedule衰到0, test-val v2 选卡
# peak lr=8e-6(prod 0.4x) steps=2000 (~9.5h) — 主力压榨
# 地板=seed2 73.69, 任何时候不如它就交 seed2
cd /workspace && unset WDS_DIR
ATT=0
until RESUME=""; [ -f outputs/jax_anneal_b/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/jax_final_s2/train_params_best.npz \
  --augment --accum 32 \
  --lr 8e-6 --proj-lr 2e-4 --vision-lr 8e-6 \
  --warmup 100 --lr-schedule linear \
  --steps 2000 --eval-every 200 --early-stop-patience 5 \
  --ckpt-every 250 \
  --seed 3 --mu-dtype float32 --prefetch-workers 24 \
  --out outputs/jax_anneal_b; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 8 ] && exit 1
  sleep 60
done
INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  outputs/jax_anneal_b/eval_preds outputs/jax_anneal_b/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds outputs/jax_anneal_b/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee outputs/jax_anneal_b/eval_report.txt
