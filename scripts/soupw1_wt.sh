#!/bin/bash
# 抬余量主实验(0812 用户"不能低于指标线"+盘点未验杠杆):
#   稀有类加权(100k 续训实测 +0.37 SubKS/+1.67 安全召回)首次打到冠军
#   soupw1(校准 81.17)身上,目标把未压底子推向 ~81.5,给 token/权重
#   压缩扩预算。配方=soupw1 暖启 + sw_rare_700k + v1 增强(无压缩开关)。
cd /workspace && unset WDS_DIR
O=outputs/soupw1_wt
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --sample-weights /data/sw_rare_700k.json \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/soupw1/soupw1.npz \
  --augment --accum 32 \
  --steps 1500 --eval-every 100 --early-stop-patience 3 \
  --ckpt-every 400 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[soupw1_wt] 无产物"; exit 1; }

INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[soupw1_wt] 完成 $(date)"
