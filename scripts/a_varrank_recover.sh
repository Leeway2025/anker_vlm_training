#!/bin/bash
# A机核心实验(用户 0810 拍板):变秩(非均匀 rank-map)+ 恢复,汤 vs 非汤各一发,
# 看恢复能到什么程度。恢复=普通SFT续训(变秩学生暂不能用蒸馏,见 train_sft L356)。
#   ① 非汤: anneal_b → map169 变秩 → SFT恢复 → 评测+校准   ← 主候选(单模谱瘦,应更耐压)
#   ② 汤  : soup     → map169 变秩 → SFT恢复 → 评测+校准   ← 对照
# 目标线 RT>=87.91 / SubKS>=80.42(校准口径),体积 ~169MB。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
SPEC="llm_attn=50,llm_mlp=50,vision_attn=62,vision_mlp=62"
ts() { date '+%m-%d %H:%M'; }

gen() {  # gen <目录> <输入npz> <SOURCE名>
  [ -f $M/$1/model.npz ] && return 0
  mkdir -p $M/$1
  echo "[$(ts)] 生成变秩 $1 ($SPEC)"
  python3 jax_impl/svd_truncate_lora.py --in "$2" --rank-map "$SPEC" \
    --act-stats $M/act_stats.npz --out $M/$1/model.npz || return 1
  echo "$3" > $M/$1/SOURCE
}

recover() {  # recover <学生目录> <输出目录> <seed>
  local IN=$M/$1/model.npz O=$M/$2 SEED=$3
  [ -f "$IN" ] || { echo "[$(ts)] $1 无产物,跳过"; return; }
  [ -f $O/eval_report.txt ] && { echo "[$(ts)] $2 已评测,跳过"; return; }
  echo "[$(ts)] === 变秩SFT恢复 $2 (学生=$1, 500步) ==="
  local ATT=0
  until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
    python jax_impl/train_sft.py $RESUME \
    --labels /data/labels_train_plus_testval_v2.jsonl \
    --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
    --rank-scheme map --train-vision --train-projector \
    --init-npz "$IN" \
    --augment --accum 32 \
    --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
    --warmup 50 --lr-schedule linear \
    --steps 500 --eval-every 100 --early-stop-patience 4 --ckpt-every 200 \
    --seed $SEED --mu-dtype float32 --prefetch-workers 24 --out $O; do
    ATT=$((ATT+1)); [ $ATT -ge 5 ] && break; sleep 60
  done
  if [ -f $O/train_params_best.npz ]; then
    INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
      /data/labels_test.jsonl /data/hf_layout.json \
      $O/eval_preds $O/train_params_best.npz 8 && \
    python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
      --labels /data/labels_test.jsonl | tee $O/eval_report.txt && \
    python3 $M/apply_calibration.py $O/eval_preds.jsonl \
      --gold /data/labels_test.jsonl | tee $O/calibrated_report.txt
    gcloud storage cp $O/eval_report.txt $O/calibrated_report.txt \
      gs://zx_vlm_dataset/backup_amachine_0810/$2/ 2>/dev/null || true
  fi
  echo "[$(ts)] $2 完成: $(tail -1 $O/calibrated_report.txt 2>/dev/null)"
}

# 等遗留 SVD/训练收尾释放 TPU
while pgrep -af 'svd_truncate|train_sft' 2>/dev/null | grep -qv 'a_varrank'; do
  echo "[$(ts)] 等遗留 TPU 进程收尾"; sleep 60
done

gen b_map169 outputs/anneal_b_best.npz anneal_b
gen map169   $M/model.npz              soup

echo "[$(ts)] ① 非汤变秩恢复(主候选)"
recover b_map169 recover_b_map169 21
echo "[$(ts)] ② 汤变秩恢复(对照)"
recover map169   recover_map169   22

echo "[$(ts)] ===== A机变秩恢复汇总(校准口径)====="
for d in recover_b_map169 recover_map169; do
  [ -f $M/$d/calibrated_report.txt ] && echo "  $d: $(tail -1 $M/$d/calibrated_report.txt)"
done
echo "[$(ts)] A机核心实验完成"
