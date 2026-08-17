#!/bin/bash
# 100k 增强试验 · B机=加权臂(处理组)。用户 0811 07:20:增强先100k试再700k。
#   与 A机 enh100k_base 完全同数据/同 init(anneal_b_best)/同超参/同 seed,
#   唯一差异:--sample-weights /data/sw_rare_100k.json(尾部 SubKS 类 hard-mining 复制,
#   qrunj 安全类 x3、其余尾类 o/s/t x2、余按 sqrt 逆频,CAP4)。隔离加权效应。
#   只训练+推理+裸评测;校准由 A 机集中对两臂统一做(回收本臂 eval_preds)。
cd /workspace && unset WDS_DIR
O=outputs/enh100k_wt
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_100k_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --sample-weights /data/sw_rare_100k.json \
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
[ -f $O/train_params_best.npz ] || { echo "[enh100k_wt] 无产物"; exit 1; }

INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[enh100k_wt] 完成 $(date)"
