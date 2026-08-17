#!/bin/bash
# 清洗夜·增强臂(B机): 与软化臂唯一差异 = --augment-v2(S2b2 消融的"开"臂)。
# B机独立跑, 无锁竞争。B机代码需先 git pull 到 3ba1380+(含 augment-v2/断点)。
set -e
cd ~/code/anker_vlm_training
git pull --ff-only 2>/dev/null || true
ts() { date '+%m-%d %H:%M:%S'; }
OUT=outputs/jax_5b_soft_v2

echo "[$(ts)] 增强臂训练开始(B机首跑, 含基座权重 gs:// 首次拉取)"
ATT=0
until sudo docker exec tpu_train bash -c "cd /workspace && unset WDS_DIR && \
  python jax_impl/train_sft.py --labels /data/labels_dedup_softclean.jsonl \
    --sample-weights /data/suspect_weights.json \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --augment-v2 --early-stop-patience 3 \
    --accum 32 --steps 1500 --eval-every 100 --val-ids /data/val_ids_v2.txt \
    --seed 1 --mu-dtype float32 --prefetch-workers 24 \
    --cartography --ckpt-every 200 --resume --out $OUT"; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && { echo "[watchdog] 连败5次放弃"; exit 1; }
  echo "[watchdog] 非零退出(第${ATT}次), 60s 后断点续跑"; sleep 60
done

echo "[$(ts)] 推理+双口径评测"
sudo docker exec tpu_train bash -c "cd /workspace && unset WDS_DIR && \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    $OUT/eval_preds $OUT/train_params_best.npz 8 && \
  python3 jax_impl/eval_metrics.py --preds $OUT/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class \
    --exclude-ids /data/test_mislabel_exclude_ids_522.txt | tee $OUT/eval_report.txt"
echo "[$(ts)] 增强臂完成。对照: A机软化臂(无v2) —— 两臂差 = v2 净效应"
