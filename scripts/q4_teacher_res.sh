#!/bin/bash
# [Q4] 现有老师在 1024(满分辨率)vs 原生 512 的自身精度对比。
# 决定能否零训练复用现有老师做满分辨率 KD 老师。纯推理,无代码改动。
# 用法: bash scripts/q4_teacher_res.sh <tag> <npz> <native_env>
#   native_env 形如 "SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=topk"
set -e
cd /workspace && unset WDS_DIR
TAG=${1:?tag}; NPZ=${2:?npz}; NATIVE=${3:?native_env}
O=outputs/q4_$TAG
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }
[ -s "$NPZ" ] || { echo "[q4 $(ts)] 缺 $NPZ"; exit 1; }

cal() { # $1=preds.jsonl $2=out.txt
  python3 outputs/class_diag.py "$1" --gold /data/labels_test.jsonl \
    --train /data/labels_train_plus_testval_v2.jsonl 2>&1 \
    | grep -iE "n=11022|RT_cal|SK_cal" | tee "$2"
}

echo "[q4 $(ts)] $TAG 原生 512 推理: env=[$NATIVE]"
env $NATIVE INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/preds_512 $NPZ 8
echo "[q4 $(ts)] $TAG 原生 512 裸评:"
python3 jax_impl/eval_metrics.py --preds $O/preds_512.jsonl --labels /data/labels_test.jsonl | tee $O/bare_512.txt
cal $O/preds_512.jsonl $O/calib_512.txt

echo "[q4 $(ts)] $TAG 满分辨率 1024 推理: SELECT_TOKENS_K=0"
SELECT_TOKENS_K=0 MAX_SOFT_TOKENS=64 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/preds_1024 $NPZ 8
echo "[q4 $(ts)] $TAG 满分辨率 1024 裸评:"
python3 jax_impl/eval_metrics.py --preds $O/preds_1024.jsonl --labels /data/labels_test.jsonl | tee $O/bare_1024.txt
cal $O/preds_1024.jsonl $O/calib_1024.txt

echo "[q4 $(ts)] ==== $TAG 老师自身精度 512 vs 1024(class_diag 同口径)===="
echo "  512:  $(cat $O/calib_512.txt 2>/dev/null | tr '\n' ' ')"
echo "  1024: $(cat $O/calib_1024.txt 2>/dev/null | tr '\n' ' ')"
echo "[q4 $(ts)] 完成 $TAG"
