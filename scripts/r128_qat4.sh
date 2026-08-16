#!/bin/bash
# r128 int4-QAT 疗伤:从 r128 fp32(80.63)出发,前向插 int4 假量化(STE),
# 双师KD(81.03)+ 稀有类加权,让权重学会适应 4bit 舍入,把 int4 从 80.28 拉回 ≥80.42。
# LoRA-only 假量化(bulk 241M);proj/vision 冻结(与 80.28 测法同口径,部署照 int4)。
# 口径:topk K=32(与 r128 fp32 80.63 / int4 80.28 一致,非 dyn)。
set -e
cd /workspace && unset WDS_DIR
export SELECT_TOKENS_K=32
export QAT_INT4=1
ts() { date '+%m-%d %H:%M'; }
echo "[$(ts)] r128 int4-QAT 起(QAT_INT4=1, topk K32)"
python3 jax_impl/train_sft.py \
  --labels /data/labels_train_plus_testval_v2.jsonl --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 128 \
  --init-npz outputs/soup_tk32_r128/model.npz \
  --teacher-npz outputs/soup_size/teacher1_tksel32b_u512.npz \
  --teacher-npz2 outputs/soup_size/teacher2_hyb2b_u512.npz \
  --distill-coef 0.5 --distill-temp 2.0 \
  --sample-weights /data/sw_rare_700k.json \
  --augment --accum 16 --lr 1e-5 \
  --loraplus-ratio 1 --warmup 30 --lr-schedule linear \
  --steps 300 --eval-every 50 --early-stop-patience 4 --ckpt-every 100 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out outputs/r128_qat4
echo "[$(ts)] QAT 训练结束,量化 best → int4 并全量评测"
O=outputs/r128_qat4
R=outputs/soup_tk32_r128
python3 outputs/delivery_0807/quantize_lora.py --in $O/train_params_best.npz \
  --out-int4 $O/model_int4_dequant.npz 2>&1 | tail -2
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/qeval_int4 $O/model_int4_dequant.npz 8
python3 jax_impl/eval_metrics.py --preds $O/qeval_int4.jsonl \
  --labels /data/labels_test.jsonl | tee $O/qeval_int4_report.txt
python3 outputs/class_diag.py $O/qeval_int4.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib_int4.txt
echo "[$(ts)] ===== r128 int4-QAT 汇总(线 87.91/80.42,预算 121.7MB)====="
echo "  r128 fp32(源)  : $(cat $R/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  r128 int4(PTQ) : $(cat $R/fair_calib_int4.txt 2>/dev/null | tr '\n' ' ')"
echo "  r128 int4(QAT) : $(cat $O/fair_calib_int4.txt 2>/dev/null | tr '\n' ' ')"
echo "[$(ts)] r128_qat4 完成"
