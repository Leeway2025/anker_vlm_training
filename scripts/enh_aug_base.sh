#!/bin/bash
# 增强消融 · A机=基线臂(v1 增强,from-scratch)。用户 0811 07:46:增强=数据增强。
#   遵代码红线:慢显性增强"增量续训验证必假阴性"→必须 from-scratch 消融。
#   复刻 seed2 基座配方(投影暖启 proj_a + prod秩 + augment + accum32 + 1500步 +
#   eval100 + 早停3 + 固定val),与 B机 enh_aug_v3 配对,唯一差异=B多 --augment-v3。
#   只训练+推理+裸评测;校准在 A 机集中对两臂统一做(拉 B 的 eval_preds)。
cd /workspace && unset WDS_DIR
O=outputs/enh_aug_base
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_100k_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/jax_5a/proj_a.npz \
  --augment --accum 32 \
  --steps 1500 --eval-every 100 --early-stop-patience 3 \
  --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[enh_aug_base] 无产物"; exit 1; }

INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[enh_aug_base] 完成 $(date)"
