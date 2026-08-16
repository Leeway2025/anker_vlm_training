#!/bin/bash
# 客户 0815 "按这个方法试,目标 r64 int8 在 input 减半(K=32)下达标"。
# = 冠军 ens_r64_dyn 配方(dyn K=32 + uniform r64 + 双师合议KD + sw_rare + augment)
#   完全同配,唯一差别 CAP_WEIGHT=0(caption 段损失权重归零,梯度全压到 RT|分隔符|SubKS
#   打分位)。基线冠军 int8=80.16 / fp32=80.12,线=87.91/80.42,只差 +0.26。
# 判据:int8 校准 SubKS 过 80.42 → r64 int8 122MB 档达标定档;≈80.16 → caption-mask 无增益。
set -e
cd /workspace && unset WDS_DIR
T=outputs/soup_size
O=outputs/capmask_r64_dyn
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }

for f in $T/student_r64.npz $T/teacher1_tksel32b_u512.npz $T/teacher2_hyb2b_u512.npz; do
  [ -s "$f" ] || { echo "[capmask_r64 $(ts)] 缺 $f"; exit 1; }
done

echo "[capmask_r64 $(ts)] 等 vfio 真空 …"
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

echo "[capmask_r64 $(ts)] 起训:冠军同配 + CAP_WEIGHT=0(只训打分位),dyn K=32 uniform r64 双师KD sw_rare 500步"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  CAP_WEIGHT=0 SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --init-npz $T/student_r64.npz \
  --teacher-npz $T/teacher1_tksel32b_u512.npz \
  --teacher-npz2 $T/teacher2_hyb2b_u512.npz \
  --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector \
  --sample-weights /data/sw_rare_700k.json \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear \
  --steps 500 --eval-every 50 --early-stop-patience 4 --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); echo "[retry] $ATT $(ts)"; [ $ATT -ge 10 ] && exit 1; sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[capmask_r64] 无产物"; exit 1; }

# --- fp32 dyn 推理 + 裸评 ---
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt

# --- int8 量化 + dyn 推理 + 裸评(交付口径)---
python3 outputs/delivery_0807/quantize_lora.py --in $O/train_params_best.npz \
  --out-bf16 $O/model_bf16.npz --out-int8 $O/model_int8sim.npz
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds_int8 $O/model_int8sim.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report_int8.txt

# --- class_diag 公平校准(与线同口径)---
calib() {
  [ -s "$1" ] || { echo "[capmask64-calib $(ts)] 缺 $1"; return 1; }
  echo "[capmask64-calib $(ts)] class_diag $3 ($(wc -l < $1) 行)"
  python3 outputs/class_diag.py "$1" \
    --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}
calib $O/eval_preds.jsonl      $O/fair_calib_fp32.txt "fp32"
calib $O/eval_preds_int8.jsonl $O/fair_calib_int8.txt "int8"

echo "[capmask_r64 $(ts)] ==== r64+dyn+双师KD + caption-mask(CAP_WEIGHT=0) 公平校准 ===="
echo "  fp32: $(cat $O/fair_calib_fp32.txt 2>/dev/null | tr '\n' ' ')"
echo "  int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: 线 87.91/80.42 ; 冠军 ens_r64_dyn int8=80.16/fp32=80.12(同配 CAP_WEIGHT=1)"
echo "  判据: int8 SubKS 过 80.42 → r64 int8 K=32 达标定档; ≈80.16 → caption-mask 无增益"
echo "[capmask_r64 完成 $(ts)]"
