#!/usr/bin/env bash
# 归因显著性 oracle 决定性实验(棒一):
#   1. 用 teacher(soup_tk32 prod, fp32=88.11/80.84)对全测试集抽 token 显著性
#   2. 在可交付 r64 底座上,dyn K=32 但用『显著性』替掉运动/范数选择 → 推理
#   3. eval + class_diag 公平校准,对线 80.42 / 对基线 dyn(motion)=80.16
# 判据:显著性选择 >80.16(尤其奔 80.42)→ 选择杠杆活,值得用归因重训头;
#       ≈80.16 → 选择到顶,如实上报并转容量方案。
set -e
cd "$(dirname "$0")/.."
ts() { date '+%m-%d %H:%M'; }
SAL_INIT=${1:-outputs/soup_tk32/train_params_best.npz}
SAL_SCHEME=${2:-prod}
STUDENT=${3:-outputs/ens_r64_dyn/train_params_best.npz}
TAG=${4:-teacher}
SAL=outputs/sal/${TAG}
O=outputs/oracle_${TAG}
mkdir -p "$O"

echo "[oracle $(ts)] 等 vfio 真空 …"
while :; do
  hold=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    comm=$(cat /proc/$p/comm 2>/dev/null) || true
    case "$comm" in python|python3) : ;; *) continue;; esac
    if ls -l /proc/$p/fd 2>/dev/null | grep -q vfio; then hold=1; break; fi
  done
  [ "$hold" = 0 ] && break
  sleep 15
done

if [ ! -s "${SAL}.npz" ]; then
  echo "[oracle $(ts)] 阶段1 抽显著性(init=$SAL_INIT scheme=$SAL_SCHEME)"
  bash scripts/sal_sharded.sh "$SAL_INIT" "$SAL_SCHEME" "$SAL"
else
  echo "[oracle $(ts)] 复用已存在 ${SAL}.npz"
fi

echo "[oracle $(ts)] 阶段2 oracle 推理(student=$STUDENT, dyn K32 + 显著性选择)"
rm -f $O/eval_preds*.jsonl 2>/dev/null || true
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn \
INFER_ARGS="--ext-score-npz ${SAL}.npz --dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python3 /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds "$STUDENT" 8

echo "[oracle $(ts)] 阶段3 eval + class_diag"
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt

echo "[oracle $(ts)] ==== 归因显著性 oracle 汇总(${TAG}) ===="
echo "  校准: $(cat $O/fair_calib.txt | tr '\n' ' ')"
echo "  对照: dyn(motion/norm) 基线 int8=80.16 / fp32≈80.12 ; 线 87.91/80.42"
echo "[oracle 完成 $(ts)]"
