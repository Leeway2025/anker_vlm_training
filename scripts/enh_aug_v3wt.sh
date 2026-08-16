#!/bin/bash
# 增强+加权 叠加确认臂(用户 0811 "增强+加权不也能加分吗" + "尽量排满")。
#   与 enh_aug_v3 完全同配方(from-scratch/proj_a/prod秩/v1+v3增强/1500步/seed7),
#   唯一差异 = 多 --sample-weights /data/sw_rare_100k.json → 与单独 v3、单独加权同口径,
#   直接量出叠加是加性/协同/递减,给 700k 权重×增强强度定调。
#   产物 outputs/enh_aug_v3wt;跑完推理+裸评测,校准集中在 A 机做。
cd /workspace && unset WDS_DIR
O=outputs/enh_aug_v3wt
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_100k_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --sample-weights /data/sw_rare_100k.json \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/jax_5a/proj_a.npz \
  --augment --augment-v3 --accum 32 \
  --steps 1500 --eval-every 100 --early-stop-patience 3 \
  --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[enh_aug_v3wt] 无产物"; exit 1; }

INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[enh_aug_v3wt] 完成 $(date)"
