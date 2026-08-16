#!/bin/bash
# [EXP-FR1] 满分辨率老师蒸馏(A 机,单老师)——损失侧到顶(cb_pair 80.29)后的唯一正交轴。
# 与 cb_pair 配方一字不改,只改两处:
#   1) 老师换成 soupw1_u512(满 token 训练的 prod 汤,lossless pad 到 uniform512,满 token 自评 ~81.17
#      >> u512 老师的 80.71)——把更富的 logit 蒸进 512-token 学生。
#   2) KD_TEACHER_FULLRES=1:老师前向走满 1024 token(不压),学生仍压 512 → 解封 KD 信息上界。
# 学生前向/推理/交付字节格式全不变(full_res 仅作用于训练期老师前向)→ RK 安全。
# 判据:int8 + class_diag 同口径。线 87.91/80.42;cb_pair(u512老师)80.29;冠军 80.16。
set -e
cd /workspace && unset WDS_DIR
T=outputs/soup_size
O=outputs/cb_fullres_r64
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }

for f in $T/student_r64.npz $T/soupw1_u512.npz outputs/confuse_sw/pair_normal.json; do
  [ -s "$f" ] || { echo "[fr1 $(ts)] 缺 $f"; exit 1; }
done

echo "[fr1 $(ts)] 等 vfio 真空 …"
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

echo "[fr1 $(ts)] 起训(满分辨率单老师 soupw1_u512, KD_TEACHER_FULLRES=1, dyn K=32 学生, uniform r64, pair_normal 加权, FOCAL_GAMMA=1.5, 500步 eval50 早停4)"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn FOCAL_GAMMA=1.5 KD_TEACHER_FULLRES=1 \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --init-npz $T/student_r64.npz \
  --teacher-npz $T/soupw1_u512.npz \
  --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector \
  --sample-weights outputs/confuse_sw/pair_normal.json \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear \
  --steps 500 --eval-every 50 --early-stop-patience 4 --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); echo "[retry] $ATT $(ts)"; [ $ATT -ge 10 ] && exit 1; sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[fr1] 无产物"; exit 1; }

# --- fp32 dyn 推理(学生 K=32,无 full_res)+ 裸评 ---
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt

# --- int8 量化 + dyn 推理 + 裸评 ---
python3 outputs/delivery_0807/quantize_lora.py --in $O/train_params_best.npz \
  --out-bf16 $O/model_bf16.npz --out-int8 $O/model_int8sim.npz
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds_int8 $O/model_int8sim.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report_int8.txt

# --- class_diag 公平校准(与线同口径,CPU)---
calib() {
  [ -s "$1" ] || { echo "[fr1-calib $(ts)] 缺 $1"; return 1; }
  echo "[fr1-calib $(ts)] class_diag $3 ($(wc -l < $1) 行)"
  python3 outputs/class_diag.py "$1" \
    --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}
calib $O/eval_preds.jsonl      $O/fair_calib_fp32.txt "fp32"
calib $O/eval_preds_int8.jsonl $O/fair_calib_int8.txt "int8"

echo "[fr1 $(ts)] ==== 满分辨率单老师蒸馏 公平校准汇总 ===="
echo "  fp32: $(cat $O/fair_calib_fp32.txt 2>/dev/null | tr '\n' ' ')"
echo "  int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: 线 87.91/80.42 ; cb_pair(u512老师)int8=87.71/80.29 ; 冠军 80.16"
echo "[fr1 完成 $(ts)]"
