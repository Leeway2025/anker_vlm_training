#!/bin/bash
# r64 dyn-aware 适配:训练时即用 dyn 动态分配(消除训/推错配),从已蒸馏 student_r64 续,
# KD from teacher_u512(coef0.5/temp2.0)+ 稀有类加权。目标=把 r64+dyn 零样本 87.56/80.12
# 的剩余 ~0.35RT/0.30SK 补过线(122MB int8 交付候选)。配方复刻 soup_size/distill + dyn env。
set -e
cd /workspace && unset WDS_DIR
O=outputs/r64dyn_adapt
ts() { date '+%m-%d %H:%M'; }

echo "[r64dyn-adapt $(ts)] 等 vfio 真空 …"
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

echo "[r64dyn-adapt $(ts)] 训练(dyn K=32, uniform r64, KD u512, sw_rare, 400步 eval50 早停4)"
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn \
  python3 jax_impl/train_sft.py --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --init-npz outputs/soup_size/student_r64.npz \
  --teacher-npz outputs/soup_size/teacher_u512.npz --distill-coef 0.5 --distill-temp 2.0 \
  --train-vision --train-projector --sample-weights /data/sw_rare_700k.json --augment \
  --accum 16 --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --loraplus-ratio 1 \
  --warmup 30 --lr-schedule linear --steps 400 --eval-every 50 --early-stop-patience 4 \
  --ckpt-every 100 --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O

echo "[r64dyn-adapt $(ts)] 带 dyn 推理(8卡, --dump-letter-logits)"
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl --labels /data/labels_test.jsonl | tee $O/eval_report.txt
python3 outputs/class_diag.py $O/eval_preds.jsonl --gold /data/labels_test.jsonl \
  --train /data/labels_train_plus_testval_v2.jsonl 2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt

echo "[r64dyn-adapt $(ts)] ==== r64 dyn-aware 适配汇总 ===="
echo "  fp32校准: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  参照: r64+dyn零样本=87.56/80.12 ; r64-KD=87.68/79.68 ; 线 87.91/80.42"
echo "  下步: fp32过线→int8复评(122MB定档); 差→叠合议teacher(88.46/81.03)重蒸"
echo "[r64dyn-adapt 完成 $(ts)]"
