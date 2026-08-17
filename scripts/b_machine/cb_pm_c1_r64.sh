#!/bin/bash
# [EXP-pm-c1] pair-margin(coef=0.1,margin=2.0)叠加在 cb_pair(加权+focal1.5)之上;在 SubKS 位对同父兄弟施 hinge,补 focal 压不动的硬兄弟判别缺口
# 与冠军 ens_r64_dyn 配方一字不改(uniform r64 + dyn K=32 + 双老师KD + augment + seed7 + 500步)
# 只改两处(均纯训练期、RK 安全,交付 adapter 字节格式不变):
#   1) 样本权重 = pair_normal.json:m 的同父(Normal)兄弟 a..l 上调 2.0x(它们正是漏进 m 的
#      混淆源),m 保持 1.0(不压!这是与 cb_moderate/cb_aggr 的关键区别——那两个压 m 反伤 m 召回),
#      安全类 n-u 保持 1.0。定向治"塌进 m",而非无差别压整类。
#   2) FOCAL_GAMMA=1.5:focal 自动压低易样本(高置信 m)梯度、聚焦难兄弟。
# 完了 fp32→int8 + class_diag 公平校准(与线 87.91/80.42 同口径)。判据:int8 SubKS vs 冠军 80.16 / 线 80.42。
set -e
cd /workspace && unset WDS_DIR
T=outputs/soup_size
O=outputs/cb_pm_c1_r64
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }

for f in $T/student_r64.npz $T/teacher1_tksel32b_u512.npz $T/teacher2_hyb2b_u512.npz outputs/confuse_sw/pair_normal.json; do
  [ -s "$f" ] || { echo "[cb_pair $(ts)] 缺 $f"; exit 1; }
done

echo "[cb_pair $(ts)] 等 vfio 真空 …"
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

echo "[cb_pair $(ts)] 起训(dyn K=32, uniform r64, 双老师, pair_normal 定向加权, FOCAL_GAMMA=1.5, 500步 eval50 早停4)"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn FOCAL_GAMMA=1.5 PAIR_MARGIN_COEF=0.1 PAIR_MARGIN=2.0 \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --init-npz $T/student_r64.npz \
  --teacher-npz $T/teacher1_tksel32b_u512.npz \
  --teacher-npz2 $T/teacher2_hyb2b_u512.npz \
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
[ -f $O/train_params_best.npz ] || { echo "[cb_pair] 无产物"; exit 1; }

# --- fp32 dyn 推理(带 letter-logits 供 class_diag)+ 裸评 ---
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
  [ -s "$1" ] || { echo "[cb_pair-calib $(ts)] 缺 $1"; return 1; }
  echo "[cb_pair-calib $(ts)] class_diag $3 ($(wc -l < $1) 行)"
  python3 outputs/class_diag.py "$1" \
    --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}
calib $O/eval_preds.jsonl      $O/fair_calib_fp32.txt "fp32"
calib $O/eval_preds_int8.jsonl $O/fair_calib_int8.txt "int8"

echo "[cb_pair $(ts)] ==== 混淆对定向加权+focal 公平校准汇总 ===="
echo "  fp32: $(cat $O/fair_calib_fp32.txt 2>/dev/null | tr '\n' ' ')"
echo "  int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: 线 87.91/80.42 ; 冠军 int8=87.78/80.16 ; cb_aggr(压m)int8=87.69/79.70"
echo "[cb_pair 完成 $(ts)]"
