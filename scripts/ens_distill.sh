#!/bin/bash
# 合议双老师重蒸(0814 用户令"按这个思路"):唯一变量 = 老师从 u512 单师
# 换成 真81.03 合议对(tksel32b_u512 + hyb2b_u512),其余与 vrk32_distill 同配方
# (vr-attn best init + topk K=32 + sw_rare + augment + rank-scheme map),以隔离
# "更强老师能否把 SubKS 从 gate 的 78.42 抬过线 80.42"。完了 fp32→int8 评测 +
# class_diag 公平校准(与线 87.91/80.42 同口径),脚本自带校准、无需外挂 waiter。
cd /workspace && unset WDS_DIR
export SELECT_TOKENS_K=32          # topk/soft64,与 vrk32_distill 同口径(apples-to-apples)
M=outputs/delivery_0807
T=outputs/soup_size
O=outputs/ens_distill
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }

# 前置校验:两师就位
for f in $T/teacher1_tksel32b_u512.npz $T/teacher2_hyb2b_u512.npz \
         $M/vr_distill_attn/train_params_best.npz; do
  [ -s "$f" ] || { echo "[ens_distill $(ts)] 缺 $f"; exit 1; }
done

echo "[ens_distill $(ts)] 起合议重蒸(双老师 tksel32b_u512 + hyb2b_u512)"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme map --init-npz $M/vr_distill_attn/train_params_best.npz \
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
[ -f $O/train_params_best.npz ] || { echo "[ens_distill] 无产物"; exit 1; }

# --- fp32 K32 推理(带 letter-logits 供 class_diag)+ 裸评 ---
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt

# --- int8 量化 + K32 推理 + 裸评 ---
python3 outputs/delivery_0807/quantize_lora.py --in $O/train_params_best.npz \
  --out-bf16 $O/model_bf16.npz --out-int8 $O/model_int8sim.npz
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds_int8 $O/model_int8sim.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds_int8.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report_int8.txt

# --- class_diag 公平校准(与线同口径,CPU)---
calib() {  # $1=preds $2=out.txt $3=标签
  [ -s "$1" ] || { echo "[ens-calib $(ts)] 缺 $1"; return 1; }
  echo "[ens-calib $(ts)] class_diag $3 ($(wc -l < $1) 行)"
  python3 outputs/class_diag.py "$1" \
    --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}
calib $O/eval_preds.jsonl      $O/fair_calib_fp32.txt "fp32"
calib $O/eval_preds_int8.jsonl $O/fair_calib_int8.txt "int8"

echo "[ens_distill $(ts)] ==== 合议双老师重蒸 公平校准汇总 ===="
echo "  fp32: $(cat $O/fair_calib_fp32.txt 2>/dev/null | tr '\n' ' ')"
echo "  int8: $(cat $O/fair_calib_int8.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: 线 87.91/80.42 ; gate(u512单师)int8=87.30/78.42 ; r64+dyn=87.56/80.12 ; 合议上界=88.46/81.03"
echo "  判据: int8 过线→122MB/vr合议档定档交付; 差线→考虑 dyn口径/更多步/或最终取舍"
echo "[ens_distill 完成 $(ts)]"
