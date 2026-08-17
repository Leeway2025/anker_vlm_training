#!/bin/bash
# r96 v3(0813 终轮:v1/v2 跷跷板均未双过,最后结构牌=激活感知截断):
#   ①K=32 输入下采集激活统计(n=256)②act-stats SVD 截 r96(起点损耗更小)
#   ③KD 500 步恢复(不带 rt-w,保 SubKS;RT 靠更高起点)④评测。
cd /workspace && unset WDS_DIR
export SELECT_TOKENS_K=32
S=outputs/r96v3
mkdir -p $S
[ -f $S/act_stats.npz ] || python3 jax_impl/collect_act_stats.py \
  --ckpt outputs/soup_tk32/train_params_best.npz \
  --labels /data/labels_train_plus_testval_v2.jsonl --layout /data/hf_layout.json \
  --n 256 --out $S/act_stats.npz
[ -f $S/model.npz ] || python3 jax_impl/svd_truncate_lora.py \
  --in outputs/soup_tk32/train_params_best.npz --rank 96 \
  --act-stats $S/act_stats.npz --out $S/model.npz
ATT=0
until RESUME=""; [ -f $S/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 96 \
  --init-npz $S/model.npz \
  --teacher-npz outputs/soup_size/teacher_u512.npz \
  --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector \
  --sample-weights /data/sw_rare_700k.json \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 \
  --loraplus-ratio 1 --warmup 30 --lr-schedule linear \
  --steps 500 --eval-every 50 --early-stop-patience 4 --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $S; do
  ATT=$((ATT+1)); echo "[retry] $ATT $(date)"; [ $ATT -ge 10 ] && exit 1; sleep 60
done
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $S/eval_preds $S/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $S/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $S/eval_report.txt
echo "[r96v3] 完成 $(date)"
