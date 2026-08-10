#!/bin/bash
# 变秩夜链(A机): CPU 先生成 rank-map 折叠产物 + 单模对照,等 compress_ladder
# 让出 TPU 后,依次跑 map 回环门禁(r0)+ 各档直评。
# 门禁: trunc_r0(满秩重分解, map 路径)裸分必须 = 88.12/79.70,否则全部作废。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
ts() { date '+%m-%d %H:%M'; }

gen() {  # gen <目录> <输入npz> <rank-map spec>
  [ -f $M/$1/model.npz ] && return 0
  mkdir -p $M/$1
  echo "[$(ts)] 生成 $1: $3"
  python3 jax_impl/svd_truncate_lora.py --in "$2" \
    --rank-map "$3" --act-stats $M/act_stats.npz \
    --out $M/$1/model.npz || return 1
  ls -l --block-size=M $M/$1/model.npz
}
# 三档 rank-map(对象=交付汤)+ 单模对照(anneal_b 同 202 档配方)
gen map202    $M/model.npz            "llm_attn=63,llm_mlp=63,vision_attn=72,vision_mlp=72"
gen map169    $M/model.npz            "llm_attn=50,llm_mlp=50,vision_attn=62,vision_mlp=62"
gen mapcurve  $M/model.npz            "llm_attn=96,llm_mlp=128,vision_attn=96,vision_mlp=128"
gen b_map202  outputs/anneal_b_best.npz "llm_attn=63,llm_mlp=63,vision_attn=72,vision_mlp=72"

# ---- 等阶梯让出 TPU ----
while ! grep -q '阶梯全部完成' $M/ladder.log 2>/dev/null; do sleep 180; done
sleep 30   # libtpu 锁释放余量(踩坑清单#4)

evalone() {  # evalone <目录>
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

evalone trunc_r0        # map 回环门禁: 必须复现 88.12/79.70
if ! grep -q 'RoleType acc   = 88.1' $M/trunc_r0/eval_report.txt 2>/dev/null; then
  echo "[$(ts)] ⚠️ r0 回环门禁未对齐(见 trunc_r0/eval_report.txt)——"
  echo "         变秩(map)直评结果不可信,仅 uniform 阶梯可用"
fi
evalone map202
evalone map169
evalone mapcurve
evalone b_map202        # 单模压缩对照: 汤谱肥 vs 单模起点低,谁赢看这局

echo "[$(ts)] 夜链完成 —— 汇总(校准口径):"
for d in trunc_r0 trunc_r128 trunc_r96 trunc_r72 trunc_r64 \
         map202 map169 mapcurve b_map202; do
  [ -f $M/$d/calibrated_report.txt ] && \
    echo "  $d: $(tail -1 $M/$d/calibrated_report.txt)"
done
gcloud storage cp $M/*/eval_report.txt $M/*/calibrated_report.txt \
  gs://zx_vlm_dataset/backup_amachine_0810/ 2>/dev/null || true
