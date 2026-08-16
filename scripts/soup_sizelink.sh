#!/bin/bash
# 压缩域体积链(0813):源=当前最佳汤 soup_tk32(校准 SubKS 80.84,select K=32)。
#   ① 从 soup 建 u512 老师(pad-to-uniform,无损)+ r64 学生(激活感知 SVD 截断)
#   ② K=32 环境下 uniform-r64 蒸馏(老师=u512 soup)恢复截断损失,~300步
#   ③ fp32 评测 + 公平校准(class_diag)
#   ④ int8 逐通道量化 + 反量化评测 + 公平校准
#   判据: int8 校准后 RT>=87.91 且 SubKS>=80.42 → 体积链守线,得可交付小模型。
# 全程 SELECT_TOKENS_K=32(soup_tk32=纯选 K=32,topk 默认模式)。
set -e
cd /workspace && unset WDS_DIR
export SELECT_TOKENS_K=32
SRC=outputs/soup_tk32/train_params_best.npz
O=outputs/soup_size
DEL=outputs/delivery_0807
ACT=$DEL/act_stats.npz
ts() { date '+%m-%d %H:%M'; }
mkdir -p $O

echo "[$(ts)] ① 建 u512 老师(pad-to-uniform)"
python jax_impl/svd_truncate_lora.py --in $SRC --out $O/teacher_u512.npz \
  --rank 512 --pad-to-uniform 2>&1 | tail -3

echo "[$(ts)] ① 建 r64 学生(激活感知 SVD 截断)"
python jax_impl/svd_truncate_lora.py --in $SRC --out $O/student_r64.npz \
  --rank 64 --act-stats $ACT 2>&1 | tail -5

echo "[$(ts)] ② uniform-r64 蒸馏(老师=u512 soup)K=32  300步"
ATT=0
until RESUME=""; [ -f $O/distill/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 --init-npz $O/student_r64.npz \
  --teacher-npz $O/teacher_u512.npz --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector \
  --sample-weights /data/sw_rare_700k.json \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear \
  --steps 300 --eval-every 50 --early-stop-patience 4 --ckpt-every 100 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O/distill; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && break; sleep 60
done
[ -f $O/distill/train_params_best.npz ] || { echo "[$(ts)] 蒸馏无产物,退出"; exit 1; }

echo "[$(ts)] ③ fp32 评测(K=32,8卡;uniform-r64 学生→auto 判秩,勿加 --rank-scheme prod)"
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/distill/eval_preds $O/distill/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/distill/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/distill/eval_report.txt
echo "[$(ts)] === fp32 公平校准(class_diag)==="
python3 outputs/class_diag.py $O/distill/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/distill/fair_calib_fp32.txt

echo "[$(ts)] ④ int8 逐通道量化 + 反量化评测"
python3 $DEL/quantize_lora.py --in $O/distill/train_params_best.npz \
  --out-int8real $O/model_int8.npz --out-bf16 $O/model_bf16.npz 2>&1 | tail -3
python3 $DEL/dequant_int8.py --in $O/model_int8.npz --out $O/model_int8_dequant.npz
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/qeval_int8 $O/model_int8_dequant.npz 8
python3 jax_impl/eval_metrics.py --preds $O/qeval_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/qeval_int8_report.txt
echo "[$(ts)] === int8 公平校准(class_diag)==="
python3 outputs/class_diag.py $O/qeval_int8.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib_int8.txt

echo "[$(ts)] ===== 体积链汇总(公平校准口径,线 87.91/80.42)====="
echo "  fp32-r64: $(cat $O/distill/fair_calib_fp32.txt 2>/dev/null)"
echo "  int8-r64: $(cat $O/fair_calib_int8.txt 2>/dev/null)"
ls -la $O/model_int8.npz $O/model_bf16.npz 2>/dev/null
echo "[$(ts)] soup_sizelink 完成"
