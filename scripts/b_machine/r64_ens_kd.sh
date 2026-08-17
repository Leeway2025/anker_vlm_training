#!/bin/bash
# r64 合议KD终击(0814 用户令:合议蒸馏+dyn+r64+int8,不看重安全):
#   student=r64-KD产物暖启(79.68),双老师在线logits平均
#   (teacher_u512=soup_tk32系 + teacher2=tkhyb2b padded),
#   全程 SELECT_TOKENS_K=32 + sw加权,800步;评测时另跑 dyn 输入(+0.32已验)。
cd /workspace && unset WDS_DIR
export SELECT_TOKENS_K=32
O=outputs/r64_ens_kd
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --init-npz outputs/soup_size/distill/train_params_best.npz \
  --teacher-npz outputs/soup_size/teacher_u512.npz \
  --teacher-npz2 outputs/soup_size/teacher2_hyb2b_u512.npz \
  --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector \
  --sample-weights /data/sw_rare_700k.json \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear \
  --steps 800 --eval-every 50 --early-stop-patience 5 --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); echo "[retry] $ATT $(date)"; [ $ATT -ge 10 ] && exit 1; sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[r64ens] 无产物"; exit 1; }
# 评测1: uniform K=32
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
# 评测2: dyn 输入(交付形态候选)
TOKEN_COMPRESS_MODE=dyn SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds_dyn $O/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds_dyn.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report_dyn.txt
# int8: 对 uniform/dyn 中更优者在校准后决定,这里先量化备好
python3 outputs/delivery_0807/quantize_lora.py --in $O/train_params_best.npz \
  --out-bf16 $O/model_bf16.npz --out-int8 $O/model_int8sim.npz
TOKEN_COMPRESS_MODE=dyn SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds_int8dyn $O/model_int8sim.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds_int8dyn.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report_int8dyn.txt
echo "[r64_ens_kd] 完成 $(date)"
