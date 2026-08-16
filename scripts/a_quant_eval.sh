#!/bin/bash
# 评测 r64 的量化版(bf16 / int8),看能否守住 SubKS>=80.42。
#   bf16fp32: bf16 精度回读版(真存储242MB),前向本就bf16 → 预期 == fp32 的 80.56
#   int8sim : 逐通道 int8 反量化版(真存储122MB)→ 实测精度损失
# 各自 infer(--dump-letter-logits)→ eval_metrics → fit_calibration(公平口径)。
cd /workspace && unset WDS_DIR
M=outputs/delivery_0807
R=$M/ns_repair_r64
ts() { date '+%m-%d %H:%M'; }

eval_one() {
  TAG=$1; W=$2; SIZE=$3
  P=$R/qeval_$TAG
  echo "[$(ts)] === 评测 $TAG (真存储 $SIZE) ==="
  INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json \
    $P $W 8 || return 1
  python3 jax_impl/eval_metrics.py --preds $P.jsonl \
    --labels /data/labels_test.jsonl | tee $R/qeval_${TAG}_report.txt
  echo "[$(ts)] --- $TAG 公平校准 ---"
  python3 $M/fit_calibration.py $P.jsonl --gold /data/labels_test.jsonl \
    2>&1 | tee $R/qeval_${TAG}_calib.txt
}

eval_one bf16 $R/model_bf16fp32.npz 242MB
eval_one int8 $R/model_int8sim.npz  122MB
echo "[$(ts)] a_quant_eval 完成"
