#!/bin/bash
# 零样本探针:减帧+总预算不变路线。soupw1(未压冠军,16x64=81.17)直接在
# 8帧x64满token(FRAME_SUBSAMPLE=8 + SELECT_TOKENS_K=0 + hf_layout_8f.json)上零样本评。
# 只改帧数(16->8)、每帧仍满64、总视觉token 512 不变 → 隔离纯时序欠采样代价。
# 对照:soupw1 16x64 校准=88.37/81.17;16x32(选择)零样本=77.24。
set -e
cd /workspace && unset WDS_DIR
O=outputs/probe_8f64
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }
echo "[$(ts)] 8帧x64 零样本推理(8卡,layout=8f,FRAME_SUBSAMPLE=8,SELECT_TOKENS_K=0)"
FRAME_SUBSAMPLE=8 SELECT_TOKENS_K=0 MAX_SOFT_TOKENS=64 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout_8f.json \
  $O/eval_preds outputs/soupw1/soupw1.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[$(ts)] === 8帧x64 零样本 公平校准(class_diag)==="
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt
echo "[$(ts)] ===== 8帧x64 零样本汇总 ====="
echo "  裸: $(grep -iE "RoleType|SubKS|安全" $O/eval_report.txt 2>/dev/null | tr '\n' ' ')"
echo "  校准: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照 soupw1@16x64 校准=88.37/81.17 ; 16x32选择零样本=77.24"
echo "[$(ts)] probe_8f64 完成"
