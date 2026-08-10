#!/bin/bash
# A机夜班(合并版, 按重要度排序): 等 uniform 阶梯让出 TPU 后 —
#   r0 map回环门禁 → map202 → map169 → b_map202(单模对照)
#   → map202 变秩SFT修补(400步)+评测 → mapcurve 兜底档评测
# r64 蒸馏修补已挪 B机(night_repair_b.sh)。目标: 北京次日 10:00 前全出。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }

# 等旧夜链的孤儿 SVD 生成进程收尾(避免双写同一 npz)
while ls /proc/[0-9]*/cmdline 2>/dev/null | xargs -I{} sh -c 'tr "\0" " " < {} 2>/dev/null' | grep -q svd_truncate; do
  echo "[$(ts)] 等待遗留 SVD 生成收尾"; sleep 120
done

gen() {  # gen <目录> <输入npz> <rank-map spec>
  [ -f $M/$1/model.npz ] && return 0
  mkdir -p $M/$1
  echo "[$(ts)] 生成 $1: $3"
  python3 jax_impl/svd_truncate_lora.py --in "$2" \
    --rank-map "$3" --act-stats $M/act_stats.npz \
    --out $M/$1/model.npz
}
gen map202   $M/model.npz              "llm_attn=63,llm_mlp=63,vision_attn=72,vision_mlp=72"
gen map169   $M/model.npz              "llm_attn=50,llm_mlp=50,vision_attn=62,vision_mlp=62"
gen b_map202 outputs/anneal_b_best.npz "llm_attn=63,llm_mlp=63,vision_attn=72,vision_mlp=72"
gen mapcurve $M/model.npz              "llm_attn=96,llm_mlp=128,vision_attn=96,vision_mlp=128"

while ! grep -q '阶梯全部完成' $M/ladder.log 2>/dev/null; do sleep 180; done
sleep 30   # libtpu 锁余量

evalone() {
  local O=$M/$1
  [ -f $O/eval_report.txt ] && return 0
  [ -f $O/model.npz ] || { echo "[$(ts)] $1 无产物,跳过"; return; }
  echo "[$(ts)] === $1 推理评测 ==="
  INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json $O/eval_preds $O/model.npz 8 \
    || { echo "[$(ts)] $1 推理失败"; return; }
  python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $O/eval_report.txt
  python3 $M/apply_calibration.py $O/eval_preds.jsonl \
    --gold /data/labels_test.jsonl | tee $O/calibrated_report.txt
}

evalone trunc_r0     # map 回环门禁: 裸分必须 = 88.12/79.70
grep -q 'RoleType acc   = 88.1' $M/trunc_r0/eval_report.txt 2>/dev/null \
  || echo "[$(ts)] ⚠️ r0 门禁未对齐 —— 变秩结果全部存疑"
evalone map202
evalone map169
evalone b_map202

echo "[$(ts)] map169 变秩SFT修补(400步)—— 交付主候选@169MB档"
O=$M/repair_map169; ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme map --train-vision --train-projector \
  --init-npz $M/map169/model.npz \
  --augment --accum 32 \
  --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 50 --lr-schedule linear \
  --steps 400 --eval-every 100 --early-stop-patience 4 --ckpt-every 200 \
  --seed 9 --mu-dtype float32 --prefetch-workers 24 --out $O; do
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
fi

evalone mapcurve     # 兜底档最后跑

echo "[$(ts)] 汇总(校准口径):"
for d in trunc_r0 trunc_r128 trunc_r96 trunc_r72 trunc_r64 \
         map202 map169 b_map202 repair_map169 mapcurve; do
  [ -f $M/$d/calibrated_report.txt ] && \
    echo "  $d: $(tail -1 $M/$d/calibrated_report.txt)"
done
echo "[$(ts)] A机夜班完成"
