#!/bin/bash
# 700k 主力大跑 · 叠加已验杠杆(用户 0811 "增强先100k再700k" + "尽量排满")。
#   from-scratch(proj_a 暖启)+ prod秩 + train vision/proj + 全池 702k +
#   稀有类加权(sw_rare_700k,100k 已验 SubKS+0.37/安全召回+1.67)+
#   数据增强 v1+v3(域定向,增强消融不输即带上;v3 低风险、编码日夜/IR/畸变不变性)。
#   步数 2000 / eval200 / 早停4 / ckpt400,断点可 --resume。
#   产物 outputs/run_700k_enh;跑完推理+裸评测,校准集中在 A 机做。
cd /workspace && unset WDS_DIR
O=outputs/run_700k_enh
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --sample-weights /data/sw_rare_700k.json \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/jax_5a/proj_a.npz \
  --augment --augment-v3 --accum 32 \
  --steps 2000 --eval-every 200 --early-stop-patience 4 \
  --ckpt-every 400 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[run_700k_enh] 无产物"; exit 1; }

INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[run_700k_enh] 完成 $(date)"
