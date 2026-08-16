#!/bin/bash
# 选择器 Layer-A 零样本探针(轮询会话 0814 10:2x):TOKEN_DILATE 邻域膨胀重打分,
# 在 K=32 冠军底座 soup_tk32 上零样本重推(不训练、不改体积),测『重选32token』能否
# 补回过线差 0.26,重点看 m/o 接触类 + g(门铃/上门,基线57.0)。产 eval_report.txt 供
# 已armed 的 attn 探针链(pid 等待)接棒;并自带 class_diag 公平校准 + 逐类归因。
# 09:10 的旧尝试因 ens_r64_dyn 占卡失败(只落 .log 无 preds),本脚本 vfio-gated 重跑。
set -e
cd /workspace && unset WDS_DIR
O=outputs/tk32_dil_zero
B=outputs/soup_tk32/train_params_best.npz
ts() { date '+%m-%d %H:%M'; }
[ -s "$B" ] || { echo "[dil_zero $(ts)] 缺底座 $B"; exit 1; }

echo "[dil_zero $(ts)] 等 vfio 真空 …"
while :; do
  hold=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    comm=$(cat /proc/$p/comm 2>/dev/null) || true
    case "$comm" in python|python3) : ;; *) continue;; esac
    if ls -l /proc/$p/fd 2>/dev/null | grep -q vfio; then hold=1; break; fi
  done
  [ "$hold" = 0 ] && break
  sleep 20
done

echo "[dil_zero $(ts)] 起 TOKEN_DILATE 零样本重推(K=32, soup_tk32, prod秩)"
rm -f $O/eval_preds*.jsonl 2>/dev/null || true
TOKEN_DILATE=1 SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $B 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt

echo "[dil_zero $(ts)] class_diag 公平校准 + 逐类归因(CPU)"
JAX_PLATFORMS=cpu python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal|^  [mogl] |per_class|recall" | tee $O/fair_calib.txt

echo "[dil_zero $(ts)] ==== TOKEN_DILATE 零样本 汇总 ===="
echo "  cal: $(cat $O/fair_calib.txt 2>/dev/null | grep -iE 'n=11022|cal' | tr '\n' ' ')"
echo "  对照: soup_tk32 纯选K32 基线=88.11/80.84 ; g基线=57.0 ; 线=87.91/80.42"
echo "[dil_zero 完成 $(ts)]"
