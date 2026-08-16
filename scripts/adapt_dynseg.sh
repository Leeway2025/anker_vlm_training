#!/bin/bash
# dynseg 适配(动态分段预算路线):16 帧全保 + 总预算 512 视觉 token 按逐帧像素
# 活跃度动态分配(每帧∈[8,64])。配方 = 8f64/tksel32b 已验冠军适配完全一致
#   (soupw1 暖启 + 稀有类加权 sw_rare_700k + 低 lr),唯一差异 = 压缩轴:
#   FRAME_SUBSAMPLE=8 均匀减帧  →  TOKEN_COMPRESS_MODE=dynseg(16帧动态分段)。
# 逐帧预算走数据侧(seg_counts 经 tokens 尾部保留区 → loss_fn 取出 → DynVisionInput
#   数据侧 pytree → _selected),shard_map 签名零改动;TOKEN_COMPRESS_MODE!=dynseg
#   时全部默认关、现有模式零改动。零样本先跑 scripts/probe_dynseg.sh 定基线。
cd /workspace && unset WDS_DIR
O=outputs/adapt_dynseg
mkdir -p $O
export TOKEN_COMPRESS_MODE=dynseg DYNSEG_TOTAL=512 DYNSEG_FLOOR=8 \
  SELECT_TOKENS_K=0 MAX_SOFT_TOKENS=64
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json \
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
[ -f $O/train_params_best.npz ] || { echo "[adapt_dynseg] 无产物"; exit 1; }

# dynseg 评测 + 公平校准(16 帧动态分段)
TOKEN_COMPRESS_MODE=dynseg DYNSEG_TOTAL=512 DYNSEG_FLOOR=8 \
  SELECT_TOKENS_K=0 MAX_SOFT_TOKENS=64 \
  INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt
echo "[adapt_dynseg] ==== 汇总 ===="
echo "  裸: $(grep -iE "RoleType|SubKS|安全" $O/eval_report.txt 2>/dev/null | tr '\n' ' ')"
echo "  校准: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')  (线 RT87.91/SubKS80.42; 8x64零样本79.28)"
echo "[adapt_dynseg] 完成 $(date)"
