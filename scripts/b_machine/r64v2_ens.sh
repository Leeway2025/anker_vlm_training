#!/bin/bash
# r64 v2(0814 三硬约束下的剩余弹药①):act-stats 截断底子(r64首次)+合议KD
#   复用 r96v3 的 K=32 域激活统计;老师同前(u512+hyb2b_u512);800步。
cd /workspace && unset WDS_DIR
export SELECT_TOKENS_K=32
S=outputs/r64v2_ens
mkdir -p $S
[ -f $S/model.npz ] || python3 jax_impl/svd_truncate_lora.py \
  --in outputs/soup_tk32/train_params_best.npz --rank 64 \
  --act-stats outputs/r96v3/act_stats.npz --out $S/model.npz
ATT=0
until RESUME=""; [ -f $S/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --init-npz $S/model.npz \
  --teacher-npz outputs/soup_size/teacher_u512.npz \
  --teacher-npz2 outputs/soup_size/teacher2_hyb2b_u512.npz \
  --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector \
  --sample-weights /data/sw_rare_700k.json \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear \
  --steps 800 --eval-every 50 --early-stop-patience 5 --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $S; do
  ATT=$((ATT+1)); echo "[retry] $ATT $(date)"; [ $ATT -ge 10 ] && exit 1; sleep 60
done
[ -f $S/train_params_best.npz ] || exit 1
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $S/eval_preds $S/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $S/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $S/eval_report.txt
# 量化恢复环(弹药③): int8→dequant→150步KD→再int8(权重贴格,量化损耗收窄)
python3 outputs/delivery_0807/quantize_lora.py --in $S/train_params_best.npz \
  --out-bf16 $S/m_bf16.npz --out-int8 $S/m_int8a.npz
python3 outputs/delivery_0807/dequant_int8.py --in $S/m_int8a.npz --out $S/m_deq.npz --dtype fp32 2>/dev/null || cp $S/m_int8a.npz $S/m_deq.npz
Q=outputs/r64v2_qrec
mkdir -p $Q
python jax_impl/train_sft.py \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --init-npz $S/m_deq.npz \
  --teacher-npz outputs/soup_size/teacher_u512.npz \
  --teacher-npz2 outputs/soup_size/teacher2_hyb2b_u512.npz \
  --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector \
  --sample-weights /data/sw_rare_700k.json \
  --augment --accum 16 \
  --lr 5e-6 --proj-lr 1e-5 --vision-lr 5e-6 --loraplus-ratio 1 \
  --warmup 10 --lr-schedule linear \
  --steps 150 --eval-every 50 --early-stop-patience 3 --ckpt-every 100 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $Q || true
if [ -f $Q/train_params_best.npz ]; then
  python3 outputs/delivery_0807/quantize_lora.py --in $Q/train_params_best.npz \
    --out-bf16 $Q/m_bf16.npz --out-int8 $Q/m_int8.npz
  SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
    bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    $Q/eval_preds_int8 $Q/m_int8.npz 8
  python3 jax_impl/eval_metrics.py --preds $Q/eval_preds_int8.jsonl \
    --labels /data/labels_test.jsonl | tee $Q/eval_report_int8.txt
fi
echo "[r64v2_ens] 完成 $(date)"
