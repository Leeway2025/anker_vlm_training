#!/bin/bash
cd /workspace && unset WDS_DIR
ATT=0
until python jax_impl/train_sft.py \
  --labels /data/pool_700k_final_labels.jsonl \
  --layout /data/hf_layout.json --rank-scheme prod \
  --train-vision --train-projector \
  --init-npz outputs/jax_5a/proj_a.npz --augment \
  --early-stop-patience 4 --accum 32 --steps 5469 \
  --eval-every 250 --val-n 1657 --seed 1 --mu-dtype float32 \
  --prefetch-workers 24 --cartography --ckpt-every 250 --resume \
  --out outputs/jax_final_s1; do
    ATT=$((ATT+1)); [ $ATT -ge 8 ] && exit 1; sleep 60
done
INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  outputs/jax_final_s1/eval_preds outputs/jax_final_s1/train_params_best.npz 8 \
&& python3 jax_impl/eval_metrics.py --preds outputs/jax_final_s1/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee outputs/jax_final_s1/eval_report.txt
