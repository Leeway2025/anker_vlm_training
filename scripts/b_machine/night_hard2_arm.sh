#!/bin/bash
# 过夜臂·硬删v2(B机): 与软化v2臂单变量=软化vs硬删(同一套 v2 flag)。
# 等 assets_v2 就绪 + 31B 探针收工(共用 B 机 TPU)。
set -e
cd ~/code/anker_vlm_training
ts() { date '+%m-%d %H:%M:%S'; }
while [ ! -f "$HOME/data/assets_v2.done" ]; do sleep 60; done
# 等探针出 verdict(最多 4h, 超时视为探针已死直接开训)
W=0
until grep -q "\[verdict\]" logs/local_judge_probe.log 2>/dev/null; do
  sleep 120; W=$((W+1)); [ $W -ge 120 ] && break
done
sleep 60
OUT=outputs/jax_5b_hard2

echo "[$(ts)] 硬删v2臂开训"
ATT=0
until sudo docker exec tpu_train bash -c "cd /workspace && unset WDS_DIR && \
  python jax_impl/train_sft.py --labels /data/labels_dedup_clean_v2.jsonl \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 3 \
    --accum 32 --steps 1500 --eval-every 100 --val-ids /data/val_ids_v2.txt \
    --seed 1 --mu-dtype float32 --prefetch-workers 24 \
    --cartography --ckpt-every 200 --resume --out $OUT"; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && { echo "[watchdog] 连败5次放弃"; exit 1; }
  echo "[watchdog] 非零退出(第${ATT}次), 60s 后断点续跑"; sleep 60
done

echo "[$(ts)] 推理+评测(v3 口径)"
sudo docker exec tpu_train bash -c "cd /workspace && unset WDS_DIR && \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    $OUT/eval_preds $OUT/train_params_best.npz 8 && \
  python3 jax_impl/eval_metrics.py --preds $OUT/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class \
    --exclude-ids /data/test_mislabel_exclude_ids_rt_v3.txt | tee $OUT/eval_report.txt"
echo "[$(ts)] 硬删v2臂完成"
