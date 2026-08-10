#!/bin/bash
# A机夜班第二棒: 等 night_maps 完 → ① r64 蒸馏修补(uniform 学生+u512老师)
# → ② map202 变秩 SFT 修补(train_sft --rank-scheme map, 无老师)→ 各自评测。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }

while ! grep -q '夜链完成' $M/night_maps.log 2>/dev/null; do sleep 300; done
sleep 30

echo "[$(ts)] ① r64 蒸馏修补启动"
bash scripts/repair_distill.sh 64 500

echo "[$(ts)] ② map202 变秩 SFT 修补启动"
O=$M/repair_map202
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme map \
  --train-vision --train-projector \
  --init-npz $M/map202/model.npz \
  --augment --accum 32 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 50 --lr-schedule linear \
  --steps 400 --eval-every 100 --early-stop-patience 4 \
  --ckpt-every 200 \
  --seed 9 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && break
  sleep 60
done
if [ -f $O/train_params_best.npz ]; then
  INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json \
    $O/eval_preds $O/train_params_best.npz 8 && \
  python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $O/eval_report.txt && \
  python3 $M/apply_calibration.py $O/eval_preds.jsonl \
    --gold /data/labels_test.jsonl | tee $O/calibrated_report.txt
fi

echo "[$(ts)] A机夜班汇总(校准口径):"
for d in repair_r64 repair_map202; do
  [ -f $M/$d/calibrated_report.txt ] && \
    echo "  $d: $(tail -1 $M/$d/calibrated_report.txt)"
done
gcloud storage cp $M/repair_*/eval_report.txt $M/repair_*/calibrated_report.txt \
  gs://zx_vlm_dataset/backup_amachine_0810/ 2>/dev/null || true
echo "[$(ts)] A机夜班完成"
