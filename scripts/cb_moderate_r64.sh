#!/bin/bash
# [EXP-A] 混淆/平衡驱动加权: sqrt-inv-freq, m=0.5x 压 49.5% 巨类梯度霸占; 其余同冠军 ens_r64_dyn
# 决定性实验(122MB 档):强合议双老师(tksel32b_u512 + hyb2b_u512, 上界 88.46/81.03)
# 配到【原生扛 K32 的 r64/uniform 底座】上,并开 dyn 动态分配口径训/推一致。
# = r64dyn_adapt 配方(dyn K32 + uniform r64 + sw_rare + augment)但老师从 u512 单师
# 换成 真81.03 合议对。赌点:r64dyn_adapt 在 u512 单师下 = 79.79(负,因 u512 封顶 SubKS
# ~78.4);更高上界的合议老师应反转此负增益,把 r64+dyn 零样本 80.12 补过线 80.42。
# 完了 fp32→int8 + class_diag 公平校准(与线 87.91/80.42 同口径),自带校准无需外挂。
set -e
cd /workspace && unset WDS_DIR
T=outputs/soup_size
O=outputs/cb_moderate_r64
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }

# 前置校验:底座 + 两师就位
for f in $T/student_r64.npz $T/teacher1_tksel32b_u512.npz $T/teacher2_hyb2b_u512.npz; do
  [ -s "$f" ] || { echo "[ens_r64_dyn $(ts)] 缺 $f"; exit 1; }
done

echo "[ens_r64_dyn $(ts)] 等 vfio 真空 …"
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

echo "[ens_r64_dyn $(ts)] 起合议重蒸(dyn K=32, uniform r64, 双老师 tksel32b_u512+hyb2b_u512, sw_rare, 500步 eval50 早停4)"
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
  --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector \
  --sample-weights outputs/confuse_sw/cb_moderate.json \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear \
  --steps 500 --eval-every 50 --early-stop-patience 4 --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); echo "[retry] $ATT $(ts)"; [ $ATT -ge 10 ] && exit 1; sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[ens_r64_dyn] 无产物"; exit 1; }

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
calib() {  # $1=preds $2=out.txt $3=标签
  [ -s "$1" ] || { echo "[ens64-calib $(ts)] 缺 $1"; return 1; }
  echo "[ens64-calib $(ts)] class_diag $3 ($(wc -l < $1) 行)"
  python3 outputs/class_diag.py "$1" \
    --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}
calib $O/eval_preds.jsonl      $O/fair_calib_fp32.txt "fp32"
calib $O/eval_preds_int8.jsonl $O/fair_calib_int8.txt "int8"

echo "[ens_r64_dyn $(ts)] ==== r64+dyn+合议双老师重蒸 公平校准汇总 ===="
echo "  fp32: $(cat $O/fair_calib_fp32.txt 2>/dev/null | tr '\n' ' ')"
echo "  int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: 线 87.91/80.42 ; r64+dyn零样本=87.56/80.12 ; r64dyn(u512单师)=79.79 ; 合议(vr底座)int8=87.28/78.67 ; 合议上界=88.46/81.03"
echo "  判据: int8 SubKS 过 80.42→122MB 档定档交付; 差→最终取舍(接受略差/更大模型/vr全token带注)"
echo "[ens_r64_dyn 完成 $(ts)]"
