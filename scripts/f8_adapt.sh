#!/bin/bash
# 8帧×64 适配臂(B机,0813:零样本校准79.28仅差基线1.89,同预算碾压选择系):
#   FRAME_SUBSAMPLE=8 + hf_layout_8f.json,镜像 tksel32b 配方
#   (soupw1暖启+低lr3e-6+sw加权+1500步耐心4)。无 SELECT_TOKENS_K。
cd /workspace && unset WDS_DIR
export FRAME_SUBSAMPLE=8
O=outputs/f8_adapt
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout_8f.json \
  --val-ids /data/test_val_ids_v2.txt \
  --sample-weights /data/sw_rare_700k.json \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/soupw1/soupw1.npz \
  --lr 3e-6 --vision-lr 8e-6 --proj-lr 2e-4 \
  --augment --accum 32 \
  --steps 1500 --eval-every 100 --early-stop-patience 4 \
  --ckpt-every 400 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[f8_adapt] 无产物"; exit 1; }
FRAME_SUBSAMPLE=8 INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout_8f.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[f8_adapt] 完成 $(date)"
