#!/bin/bash
# 零样本探针:tome(时空 ToMe 双部软匹配)路线。soupw1(未压冠军,16x64=81.17)
# 直接在 16 帧 + 时空合并(1024 源 soft token → 单块 512)上零样本评。
# 保全部 16 帧(时序覆盖满片)、总视觉 token 与 8f64/16x32/dynseg 同=512 →
# 隔离"同预算下跨帧合并冗余背景 vs 均匀减帧(8x64)/ 逐帧选(16x32)/ 动态分段(dynseg)"。
#
# 关键:tome 用 16 帧基准排布(/data/hf_layout.json)作为【源与分解源】,1024
# 源 soft token 在设备侧确定式合并到 512;infer.py 走单块模板(哨兵总数=TOME_TOTAL
# =512,T 恒定=tome_T,丢帧间 delim),用 ex["tokens"] 而非静态 ds.template
# —— 已在 data.py/infer.py 内接好(TOKEN_COMPRESS_MODE=tome 门控)。
#
# 对照:soupw1 16x64 校准=88.37/81.17 ; 8x64 零样本=79.28 ; 16x32选择零样本=77.24
# 通过线(公平校准):RT>=87.91 / SubKS>=80.42
set -e
cd /workspace && unset WDS_DIR
O=outputs/probe_tome
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }
echo "[$(ts)] tome 零样本推理(8卡,layout=16f,TOME_TOTAL=512,双部软匹配单块)"
TOKEN_COMPRESS_MODE=tome TOME_TOTAL=512 \
  SELECT_TOKENS_K=0 MAX_SOFT_TOKENS=64 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds outputs/soupw1/soupw1.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[$(ts)] === tome 零样本 公平校准(class_diag)==="
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt
echo "[$(ts)] ===== tome 零样本汇总 ====="
echo "  裸: $(grep -iE "RoleType|SubKS|安全" $O/eval_report.txt 2>/dev/null | tr '\n' ' ')"
echo "  校准: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照 soupw1@16x64 校准=88.37/81.17 ; 8x64零样本=79.28 ; 16x32选择零样本=77.24"
echo "  通过线(公平校准):RT>=87.91 / SubKS>=80.42"
echo "[$(ts)] probe_tome 完成"
