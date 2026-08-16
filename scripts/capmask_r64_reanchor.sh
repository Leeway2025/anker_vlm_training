#!/bin/bash
# 方案A(客户 0815 预授权:CAP_WEIGHT=0 版过线后自动锚回 caption)。
# 暖启自"过线权重"capmask_r64_dyn/train_params_best.npz,用 CAP_WEIGHT=0.2 短微调把描述拉回
# (caption 段恢复 0.2 权重 + 双师 KD 一起锚向老师/GT;打分前缀 cls_w=4 仍主导 → SubKS 基本不掉)。
# 只跑 ~150 步、同低 LR/同 dyn K=32/r64/sw_rare。收尾判据:SubKS 仍 >80.42 且 caption 恢复。
set -e
cd /workspace && unset WDS_DIR
T=outputs/soup_size
S=outputs/capmask_r64_dyn          # 源:过线的纯打分版
O=outputs/capmask_r64_reanchor
CW=${CW:-0.2}                       # CAP_WEIGHT(默认0.2;掉线可降0.1重跑:CW=0.1 bash ...)
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }

[ -s $S/train_params_best.npz ] || { echo "[reanchor $(ts)] 缺源权重 $S/train_params_best.npz"; exit 1; }
for f in $T/teacher1_tksel32b_u512.npz $T/teacher2_hyb2b_u512.npz; do
  [ -s "$f" ] || { echo "[reanchor $(ts)] 缺 $f"; exit 1; }
done

echo "[reanchor $(ts)] 等 vfio 真空 …"
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

echo "[reanchor $(ts)] 起锚回:暖启过线权重 + CAP_WEIGHT=$CW,150步 dyn K=32 r64 双师KD sw_rare"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  CAP_WEIGHT=$CW SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --init-npz $S/train_params_best.npz \
  --teacher-npz $T/teacher1_tksel32b_u512.npz \
  --teacher-npz2 $T/teacher2_hyb2b_u512.npz \
  --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector \
  --sample-weights /data/sw_rare_700k.json \
  --augment --accum 16 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 15 --lr-schedule linear \
  --steps 150 --eval-every 30 --early-stop-patience 5 --ckpt-every 150 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); echo "[retry] $ATT $(ts)"; [ $ATT -ge 10 ] && exit 1; sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[reanchor] 无产物"; exit 1; }

# fp32 dyn 推理 + 裸评(含 caption 样例/格式合规)
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt

# int8 量化 + dyn 推理 + 裸评(交付口径)
python3 outputs/delivery_0807/quantize_lora.py --in $O/train_params_best.npz \
  --out-bf16 $O/model_bf16.npz --out-int8 $O/model_int8sim.npz
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds_int8 $O/model_int8sim.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report_int8.txt

calib() {
  [ -s "$1" ] || { echo "[reanchor-calib $(ts)] 缺 $1"; return 1; }
  python3 outputs/class_diag.py "$1" \
    --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}
calib $O/eval_preds.jsonl      $O/fair_calib_fp32.txt "fp32"
calib $O/eval_preds_int8.jsonl $O/fair_calib_int8.txt "int8"

echo "[reanchor $(ts)] ==== 方案A(CAP_WEIGHT=$CW 锚回)公平校准 ===="
echo "  纯打分版 int8: $(cat $S/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  锚回版   int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  判据: 锚回后 SubKS 仍 >80.42 且 caption 恢复 → 定终版; 掉线则 CW=0.1 重跑"
echo "  caption 抽查见 $O/eval_report_int8.txt(格式合规% + 样例)"
echo "[reanchor 完成 $(ts)]"
