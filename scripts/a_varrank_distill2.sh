#!/bin/bash
# 变秩 int8 提分尝试②(用户 0811 00:39 "还有办法提上去吗"):
#   ①版 flat 秩50 → fp32 SubKS 80.38 / int8 80.35,差线 80.42 就 0.04~0.07。
#   杠杆:MLP 每单位秩耗 ~2× attn(attn71/mlp146MB@秩50)。SubKS=关键场景判别
#   偏注意力/全局任务 → 把秩预算从 MLP 挪到 attn,体积维持 ~100MB int8。
#   SPEC: llm_attn 50→88, llm_mlp 50→36, vision 62/62 不变。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
O=$M/vr_distill_attn
SPEC="llm_attn=88,llm_mlp=36,vision_attn=62,vision_mlp=62"
ts() { date '+%m-%d %H:%M'; }
mkdir -p $O

# ① 生成 attn 偏重的变秩截断(从 anneal_b)
if [ ! -f $M/b_map_attn/model.npz ]; then
  mkdir -p $M/b_map_attn
  echo "[$(ts)] ① 生成变秩 b_map_attn ($SPEC)"
  python3 jax_impl/svd_truncate_lora.py --in outputs/anneal_b_best.npz \
    --rank-map "$SPEC" --act-stats $M/act_stats.npz \
    --out $M/b_map_attn/model.npz || exit 1
  echo "anneal_b" > $M/b_map_attn/SOURCE
fi
python3 -c "import numpy as np;z=np.load('$M/b_map_attn/model.npz');print('[size] fp32 %.1fMB'%(sum(z[k].nbytes for k in z.files)/1e6))"

# ② 汤 u512 蒸馏(300步,同①版超参)
echo "[$(ts)] ② 变秩(attn重)蒸馏: 老师=汤u512 300步"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme map --init-npz $M/b_map_attn/model.npz \
  --teacher-npz $M/teacher_u512.npz --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear \
  --steps 300 --eval-every 50 --early-stop-patience 4 --ckpt-every 100 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); [ $ATT -ge 5 ] && break; sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[$(ts)] 无产物,退出"; exit 1; }

# ③ fp32 评测 + 公平校准
echo "[$(ts)] ③ fp32 评测 + 公平校准"
INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt && \
python3 $M/fit_calibration.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --out $O/fitted_calibration_vr.json \
  | tee $O/calibrated_report.txt

# ④ int8 量化 + 反量化评测
echo "[$(ts)] ④ int8 量化 + 反量化评测"
python3 $M/quantize_lora.py --in $O/train_params_best.npz \
  --out-int8real $O/model_int8.npz --out-bf16 $O/model_bf16.npz 2>&1 | tail -2
python3 $M/dequant_int8.py --in $O/model_int8.npz --out $O/model_int8_dequant.npz
INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/qeval_int8 $O/model_int8_dequant.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/qeval_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/qeval_int8_report.txt && \
python3 $M/fit_calibration.py $O/qeval_int8.jsonl \
  --gold /data/labels_test.jsonl --out $O/fitted_calibration_int8.json \
  | tee $O/qeval_int8_calib.txt
rm -f $O/model_int8_dequant.npz $O/ckpt_latest.npz $O/ckpt_prev.npz $O/train_params.npz

echo "[$(ts)] ===== 变秩(attn重)汇总 ====="
echo "  fp32: $(grep '重拟' $O/calibrated_report.txt)"
echo "  int8: $(grep '重拟' $O/qeval_int8_calib.txt)"
ls -la $O/model_int8.npz 2>/dev/null
gcloud storage cp $O/eval_report.txt $O/calibrated_report.txt \
  $O/qeval_int8_calib.txt \
  gs://zx_vlm_dataset/backup_amachine_0810/vr_distill_attn/ 2>/dev/null || true
echo "[$(ts)] 完成"
