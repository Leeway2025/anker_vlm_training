#!/bin/bash
# 减帧适配(减帧/总预算不变路线,第一棒):8帧×64=512 视觉token,砍一半视觉编码器passes、无逐帧选token空洞。
# 配方 = tksel32b(已验冠军适配:soupw1暖启 + 稀有类加权 sw_rare_700k + 低lr)完全一致,
#   唯一差异 = 压缩轴:K=32逐帧选择  →  FRAME_SUBSAMPLE=8 + SELECT_TOKENS_K=0 + hf_layout_8f.json。
# 零样本已测:8×64=79.28(vs 同预算 16×32选择=77.24,+2.04);目标恢复过线 SubKS≥80.42。
cd /workspace && unset WDS_DIR
O=outputs/adapt_8f64
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  FRAME_SUBSAMPLE=8 SELECT_TOKENS_K=0 python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout_8f.json \
  --val-ids /data/test_val_ids_v2.txt \
  --sample-weights /data/sw_rare_700k.json \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/soupw1/soupw1.npz \
  --lr 3e-6 --vision-lr 8e-6 --proj-lr 2e-4 \
  --augment --accum 32 \
  --steps 1500 --eval-every 100 --early-stop-patience 4 \
  --ckpt-every 400 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[adapt_8f64] 无产物"; exit 1; }

# 8帧×64 评测 + 公平校准
FRAME_SUBSAMPLE=8 SELECT_TOKENS_K=0 MAX_SOFT_TOKENS=64 \
  INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout_8f.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt
echo "[adapt_8f64] ==== 汇总 ===="
echo "  裸: $(grep -iE "RoleType|SubKS|安全" $O/eval_report.txt 2>/dev/null | tr '\n' ' ')"
echo "  校准: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')  (线 RT87.91/SubKS80.42; 零样本79.28)"
echo "[adapt_8f64] 完成 $(date)"
