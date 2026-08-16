#!/bin/bash
# r64+dyn 零样本地板:量 dyn(动态分配,输入侧零成本)加在 122MB 交付档 student_r64 上到多少。
# 门控:等 chain_r96dyn_tome 完工(标记)+ vfio 真空,再上卡,避免与前一条接力撞卡。
# 交付判据:r64+dyn 零样本校准 vs 线 87.91/80.42;对照 r64-KD int8=87.62/79.80。
cd /workspace && unset WDS_DIR
ts() { date '+%m-%d %H:%M'; }
O=outputs/r64dyn_zero
mkdir -p $O

echo "[chain-r64dyn $(ts)] 等前一条接力(r96rec_dyn+tome)完工 …"
while :; do grep -q "chain-r96dyn 完成" outputs/chain_r96dyn_tome.log 2>/dev/null && break; sleep 20; done
echo "[chain-r64dyn $(ts)] 前链完工;等 vfio 真空 …"
while :; do
  hold=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    comm=$(cat /proc/$p/comm 2>/dev/null)
    case "$comm" in python|python3) : ;; *) continue;; esac
    if ls -l /proc/$p/fd 2>/dev/null | grep -q vfio; then hold=1; break; fi
  done
  [ "$hold" = 0 ] && break
  sleep 20
done

echo "[chain-r64dyn $(ts)] r64+dyn 零样本(8卡,student_r64,mode=dyn K=32/64)"
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds outputs/soup_size/student_r64.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt
echo "[chain-r64dyn $(ts)] ==== r64+dyn 零样本汇总 ===="
echo "  校准: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: r64-KD fp32=87.68/79.68 ; r64-KD int8(122MB)=87.62/79.80 ; 线 87.91/80.42"
echo "[chain-r64dyn 完成 $(ts)]"
