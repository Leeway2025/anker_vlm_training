#!/bin/bash
# 变秩 int8 蒸馏(用户 0810 拍板,交付后加码实验):
#   之前变秩(rank-map)恢复只能用普通SFT(train_sft L356 挡蒸馏)→ 封顶 ~80.30。
#   现已改代码解锁 map 学生 + uniform 汤老师蒸馏(prod_lora.teacher_rank 上下文
#   在老师前向短路 rank 覆盖,学生查表)。本脚本:
#     ① b_map169(anneal_b 变秩,392MB fp32)→ 汤 u512 蒸馏修补(300步)
#     ② 评测 + 公平重拟校准(fit_calibration)
#     ③ int8 逐通道量化(~98MB)+ 反量化评测 + 公平校准
#   目标: 若守线 RT>=87.91 / SubKS>=80.42,则 98MB < r64 的 122MB → 更优交付。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
O=$M/vr_distill_map169
ts() { date '+%m-%d %H:%M'; }
mkdir -p $O

echo "[$(ts)] ① 变秩蒸馏: 学生=b_map169(变秩)  老师=汤u512  300步"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme map --init-npz $M/b_map169/model.npz \
  --teacher-npz $M/teacher_u512.npz --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear \
  --steps 300 --eval-every 50 --early-stop-patience 4 --ckpt-every 100 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && break; sleep 60
done

[ -f $O/train_params_best.npz ] || { echo "[$(ts)] 训练无产物,退出"; exit 1; }

echo "[$(ts)] ② fp32 评测 + 公平校准"
INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt && \
python3 $M/fit_calibration.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --out $O/fitted_calibration_vr.json \
  | tee $O/calibrated_report.txt

echo "[$(ts)] ③ int8 逐通道量化 + 反量化评测"
python3 $M/quantize_lora.py --in $O/train_params_best.npz \
  --out-int8real $O/model_int8.npz --out-bf16 $O/model_bf16.npz 2>&1 | tail -3
python3 $M/dequant_int8.py --in $O/model_int8.npz --out $O/model_int8_dequant.npz
INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/qeval_int8 $O/model_int8_dequant.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/qeval_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/qeval_int8_report.txt && \
python3 $M/fit_calibration.py $O/qeval_int8.jsonl \
  --gold /data/labels_test.jsonl --out $O/fitted_calibration_int8.json \
  | tee $O/qeval_int8_calib.txt

echo "[$(ts)] ===== 变秩蒸馏汇总(公平校准口径)====="
echo "  fp32: $(grep -i '重拟\|refit' $O/calibrated_report.txt | tail -1)"
echo "  int8: $(grep -i '重拟\|refit' $O/qeval_int8_calib.txt | tail -1)"
ls -la $O/model_int8.npz $O/model_bf16.npz 2>/dev/null
gcloud storage cp $O/eval_report.txt $O/calibrated_report.txt \
  $O/qeval_int8_report.txt $O/qeval_int8_calib.txt \
  gs://zx_vlm_dataset/backup_amachine_0810/vr_distill_map169/ 2>/dev/null || true
echo "[$(ts)] 变秩 int8 蒸馏实验完成"
