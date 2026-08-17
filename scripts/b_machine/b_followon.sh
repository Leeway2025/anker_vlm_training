#!/bin/bash
# B机跟进(anneal_c 完成后):把 anneal_c 更好的单点做变秩(map169)+ 普通SFT恢复。
# 容器内无 pgrep,用 metrics.jsonl mtime 停滞(每步都写,连续~6分钟不变=已结束)判断 anneal_c 完成。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
SPEC="llm_attn=50,llm_mlp=50,vision_attn=62,vision_mlp=62"
MET=outputs/jax_anneal_c/metrics.jsonl
ts(){ date '+%m-%d %H:%M'; }
echo "[$(ts)] B跟进: 用 metrics mtime 停滞等 anneal_c 结束"
prev=-1; stall=0
while true; do
  cur=$(stat -c %Y "$MET" 2>/dev/null || echo 0)
  if [ "$cur" = "$prev" ]; then stall=$((stall+1)); else stall=0; fi
  prev=$cur
  [ "$stall" -ge 2 ] && break   # 连续2次(每150s)不变 ≈ 5分钟停滞
  sleep 150
done
echo "[$(ts)] anneal_c 已停止更新,采用其 best 单点"
SRC=outputs/jax_anneal_c/train_params_best.npz
[ -f "$SRC" ] || { echo "[$(ts)] 无 anneal_c best,退出"; exit 1; }
mkdir -p $M/c_map169
if [ ! -f $M/c_map169/model.npz ]; then
  echo "[$(ts)] 变秩 c_map169 (from anneal_c best)"
  python3 jax_impl/svd_truncate_lora.py --in "$SRC" --rank-map "$SPEC" \
    --act-stats $M/act_stats.npz --out $M/c_map169/model.npz || exit 1
fi
O=$M/recover_c_map169; ATT=0
echo "[$(ts)] === 变秩SFT恢复 recover_c_map169 (500步) ==="
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme map --train-vision --train-projector \
  --init-npz $M/c_map169/model.npz --augment --accum 32 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 50 --lr-schedule linear \
  --steps 500 --eval-every 100 --early-stop-patience 4 --ckpt-every 200 \
  --seed 31 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && break; sleep 60
done
if [ -f $O/train_params_best.npz ]; then
  INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json \
    $O/eval_preds $O/train_params_best.npz 8 && \
  python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $O/eval_report.txt && \
  python3 $M/apply_calibration.py $O/eval_preds.jsonl \
    --gold /data/labels_test.jsonl | tee $O/calibrated_report.txt
  gcloud storage cp $O/eval_report.txt $O/calibrated_report.txt \
    gs://zx_vlm_dataset/backup_bmachine_0810/recover_c_map169/ 2>/dev/null || true
fi
echo "[$(ts)] B跟进完成: $(tail -1 $O/calibrated_report.txt 2>/dev/null)"
