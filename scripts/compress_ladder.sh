#!/bin/bash
# uniform 截断阶梯(A机): ASVD激活感知截断 → 直评 + 全局校准
# 对象 = 0807交付汤 model.npz(当前最优)。口径: eval_report=未校准, calibrated_report=交付口径。
# 首段 r0 回环门禁(svd_truncate 验收纪律①): 满秩重分解必须与原版逐分对齐(对照 88.12/79.70)。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }

if [ ! -f $M/act_stats.npz ]; then
  echo "[$(ts)] 采集激活统计 n=256"
  python3 jax_impl/collect_act_stats.py --ckpt $M/model.npz \
    --labels /data/pool_700k_final_labels.jsonl --layout /data/hf_layout.json \
    --n 256 --out $M/act_stats.npz || exit 1
fi

for R in 128 96 72 64; do          # r0 门禁走 night_maps.sh 的 map 路径
  O=$M/trunc_r$R; mkdir -p $O
  echo "[$(ts)] === rank $R 截断 ==="
  [ -f $O/eval_report.txt ] && { echo "[$(ts)] r$R 已有评测,跳过"; continue; }
  if [ ! -f $O/model.npz ]; then
    python3 jax_impl/svd_truncate_lora.py --in $M/model.npz --rank $R \
      --act-stats $M/act_stats.npz --out $O/model.npz || continue
  fi
  echo "[$(ts)] rank $R 推理"
  INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json $O/eval_preds $O/model.npz 8 || continue
  python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $O/eval_report.txt
  python3 $M/apply_calibration.py $O/eval_preds.jsonl \
    --gold /data/labels_test.jsonl > $O/calibrated_report.txt
  echo "[$(ts)] rank $R 完成:"; tail -3 $O/calibrated_report.txt
done
echo "[$(ts)] 阶梯全部完成"
grep -H . $M/trunc_r*/calibrated_report.txt 2>/dev/null | tail -20
