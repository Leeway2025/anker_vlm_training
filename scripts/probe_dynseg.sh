#!/bin/bash
# 零样本探针:dynseg(动态分段预算)路线。soupw1(未压冠军,16x64=81.17)
# 直接在 16 帧 + 动态预算(总 512 视觉 token,逐帧∈[8,64],按像素活跃度分配)
# 上零样本评。保全部 16 帧(时序覆盖满片)、总预算与 8f64/16x32 同=512 →
# 隔离"同预算下动态分段 vs 均匀减帧(8x64)/ 均匀逐帧选(16x32)"的收益。
#
# 关键:dynseg 用 16 帧基准排布(/data/hf_layout.json)作为【分解源】,
# 但 infer.py 走逐样本变长模板(哨兵总数=DYNSEG_TOTAL=512,每帧预算按活跃度变);
# 该逐样本模板由 Dataset.__getitem__ 产出、infer.py 用 ex["tokens"] 而非静态
# ds.template —— 已在 data.py/infer.py 内接好(TOKEN_COMPRESS_MODE=dynseg 门控)。
#
# 对照:soupw1 16x64 校准=88.37/81.17 ; 8x64 零样本=79.28 ; 16x32选择零样本=77.24
set -e
cd /workspace && unset WDS_DIR
O=outputs/probe_dynseg
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }
echo "[$(ts)] dynseg 零样本推理(8卡,layout=16f,TOTAL=512,FLOOR=8,per-frame budget)"
TOKEN_COMPRESS_MODE=dynseg DYNSEG_TOTAL=512 DYNSEG_FLOOR=8 \
  SELECT_TOKENS_K=0 MAX_SOFT_TOKENS=64 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds outputs/soupw1/soupw1.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[$(ts)] === dynseg 零样本 公平校准(class_diag)==="
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt
echo "[$(ts)] ===== dynseg 零样本汇总 ====="
echo "  裸: $(grep -iE "RoleType|SubKS|安全" $O/eval_report.txt 2>/dev/null | tr '\n' ' ')"
echo "  校准: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照 soupw1@16x64 校准=88.37/81.17 ; 8x64零样本=79.28 ; 16x32选择零样本=77.24"
echo "[$(ts)] probe_dynseg 完成"
