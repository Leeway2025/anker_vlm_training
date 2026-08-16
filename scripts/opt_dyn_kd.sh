#!/bin/bash
# dyn 最优训练法·A 版(冠军 ens_r64_dyn=80.16 的受控增量):
#   同底座(uniform r64)+ 同双师合议KD(tksel32b_u512+hyb2b_u512,上界81.03)+ 同 sw_rare
#   新增:①KS 父类头(--ks-head, 6类正则 SubKS 空间, 推理丢弃, 免参数)
#         ②KD 去惰化:distill-coef 0.5→0.7, temp 2→3(冠军 KD 仅 +0.04, 近乎失效)
#         ③RT 已解 → rt-w 0 不占容量
#   从 student_r64 干净起(与冠军同起点, 便于后续种子 soup)。完了 fp32→int8→class_diag。
# 判据:int8 SubKS > 80.16(冠军)→ 有效, 再补 2 种子做 soup 冲 80.42; ≈80.16→ 该杠杆也到顶。
set -e
cd /workspace && unset WDS_DIR
T=outputs/soup_size
O=outputs/opt_dyn_kd
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }

for f in $T/student_r64.npz $T/teacher1_tksel32b_u512.npz $T/teacher2_hyb2b_u512.npz; do
  [ -s "$f" ] || { echo "[opt_dyn_kd $(ts)] 缺 $f"; exit 1; }
done

echo "[opt_dyn_kd $(ts)] 等 vfio 真空 …"
while :; do
  hold=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    comm=$(cat /proc/$p/comm 2>/dev/null) || true
    case "$comm" in python|python3) : ;; *) continue;; esac
    if ls -l /proc/$p/fd 2>/dev/null | grep -q vfio; then hold=1; break; fi
  done
  [ "$hold" = 0 ] && break
  sleep 20
done

echo "[opt_dyn_kd $(ts)] 起训(uniform r64, dyn K32, 双师, +KS头, KD coef0.7/temp3, rt-w0, 500步)"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --init-npz $T/student_r64.npz \
  --teacher-npz $T/teacher1_tksel32b_u512.npz \
  --teacher-npz2 $T/teacher2_hyb2b_u512.npz \
  --distill-coef 0.7 --distill-temp 3.0 \
  --ks-head --ks-coef 0.2 --rt-w 0 \
  --train-vision --train-projector \
  --sample-weights /data/sw_rare_700k.json \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear \
  --steps 500 --eval-every 50 --early-stop-patience 4 --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); echo "[retry] $ATT $(ts)"; [ $ATT -ge 10 ] && exit 1; sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[opt_dyn_kd] 无产物"; exit 1; }

# fp32 dyn 推理 + 裸评
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt

# int8 量化 + dyn 推理 + 裸评
python3 outputs/delivery_0807/quantize_lora.py --in $O/train_params_best.npz \
  --out-bf16 $O/model_bf16.npz --out-int8 $O/model_int8sim.npz
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds_int8 $O/model_int8sim.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report_int8.txt

# class_diag 公平校准
calib() {
  [ -s "$1" ] || { echo "[opt-calib $(ts)] 缺 $1"; return 1; }
  python3 outputs/class_diag.py "$1" \
    --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}
calib $O/eval_preds.jsonl      $O/fair_calib_fp32.txt
calib $O/eval_preds_int8.jsonl $O/fair_calib_int8.txt

echo "[opt_dyn_kd $(ts)] ==== A版(+KS头+KD去惰化) 公平校准汇总 ===="
echo "  fp32: $(cat $O/fair_calib_fp32.txt 2>/dev/null | tr '\n' ' ')"
echo "  int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: 冠军 ens_r64_dyn int8=80.16 ; 线 87.91/80.42"
echo "[opt_dyn_kd 完成 $(ts)]"
