#!/bin/bash
# B机续退火(第二段): 从 anneal_b_best 起, 更低峰值短schedule, 训到早停触发
# 目的: 兑现0807报告"继续拟合至收敛 +~0.5"; 地板=anneal_b(裸口径 RT 87.84/SubKS 79.34),
# 不如它就交 anneal_b。产物跑完立即回传 GCS(B机系统盘用完即回收)。
cd /workspace && unset WDS_DIR
ATT=0
until RESUME=""; [ -f outputs/jax_anneal_c/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/anneal_b_best.npz \
  --augment --accum 32 \
  --lr 4e-6 --proj-lr 1e-4 --vision-lr 4e-6 \
  --warmup 100 --lr-schedule linear \
  --steps 1500 --eval-every 200 --early-stop-patience 5 \
  --ckpt-every 250 \
  --seed 5 --mu-dtype float32 --prefetch-workers 24 \
  --out outputs/jax_anneal_c; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 8 ] && exit 1
  sleep 60
done
INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  outputs/jax_anneal_c/eval_preds outputs/jax_anneal_c/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds outputs/jax_anneal_c/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee outputs/jax_anneal_c/eval_report.txt
# 产物立即回 GCS(B机盘会回收)
gcloud storage cp outputs/jax_anneal_c/train_params_best.npz \
  gs://zx_vlm_dataset/backup_bmachine_0810/anneal_c_best.npz 2>/dev/null || \
  echo "[warn] GCS 回传失败, 手动补: outputs/jax_anneal_c/train_params_best.npz"
gcloud storage cp outputs/jax_anneal_c/eval_report.txt outputs/jax_anneal_c/eval_preds.jsonl \
  outputs/jax_anneal_c/metrics.jsonl gs://zx_vlm_dataset/backup_bmachine_0810/ 2>/dev/null
echo "[anneal_c] 全链完成 $(date)"
