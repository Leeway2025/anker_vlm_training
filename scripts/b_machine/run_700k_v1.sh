#!/bin/bash
# 700k 分歧成员 · v1-only(无 v3)+ 稀有类加权,seed 2(B机)。
#   与 A 机 run_700k_enh 同池同权重同超参,两点差异:① 去掉 --augment-v3(仅 v1 增强)
#   ② seed 2(A 为 seed7)。目的三合一:
#     (a) 700k 尺度上的 base(v1) vs v3 对照——在全池验证 v3 决策(100k 上 v3 −0.33SubKS/+2.88安全召回);
#     (b) 造真正有分歧的集成成员(不同增强分布+不同种子)→ 路线1 合议→蒸馏回单模的素材;
#     (c) 若 v3 在 700k 尺度反伤,这条即备用交付。
#   产物 outputs/run_700k_v1;跑完推理+裸评测。
cd /workspace && unset WDS_DIR
O=outputs/run_700k_v1
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
  --augment --accum 32 \
  --steps 2000 --eval-every 200 --early-stop-patience 4 \
  --ckpt-every 400 \
  --seed 2 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[run_700k_v1] 无产物"; exit 1; }

INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[run_700k_v1] 完成 $(date)"
