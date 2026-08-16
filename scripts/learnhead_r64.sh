#!/bin/bash
# ①学习打分头 · 第一棒(head-only,轮询会话 0814):在 r64+dyn 最佳底座
# (ens_r64_dyn best = int8 80.16,我们最好的 122MB)上加可学习打分头,warm-start
# 零init(=no-op,起点严格==底座),只训头(其余组 lr=0 冻结、但仍从 init-npz 载入
# 保底),隔离验证『学出来的选token』能否把 SubKS 抬过 80.42。任务CE损失(不带老师,
# TOKEN_LEARN_SCORE 暂不支持蒸馏)。dyn K=32 训推一致。只训练+存档;infer/int8/校准
# 待 infer.py 注入补丁落地后单独跑。
set -e
cd /workspace && unset WDS_DIR
T=outputs/ens_r64_dyn/train_params_best.npz
O=outputs/learnhead_r64
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }
[ -s "$T" ] || { echo "[learnhead $(ts)] 缺底座 $T"; exit 1; }

echo "[learnhead $(ts)] 等 vfio 真空 …"
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

echo "[learnhead $(ts)] 起 head-only 训练(TOKEN_LEARN_SCORE=1, dyn K=32, 头lr=1e-3, 其余组冻结)"
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  TOKEN_LEARN_SCORE=1 TOKEN_LEARN_RANK=16 TOKEN_STE_TEMP=1.0 TOKEN_LEARN_GAIN=8 \
  SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --init-npz $T \
  --train-vision --train-projector \
  --sample-weights /data/sw_rare_700k.json \
  --augment --accum 16 \
  --lr 0 --proj-lr 0 --vision-lr 0 --tok-lr 3e-2 --loraplus-ratio 1 \
  --warmup 20 --lr-schedule linear \
  --steps 400 --eval-every 50 --early-stop-patience 4 --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 --out $O; do
  ATT=$((ATT+1)); echo "[retry] $ATT $(ts)"; [ $ATT -ge 10 ] && exit 1; sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[learnhead] 无产物"; exit 1; }
echo "[learnhead $(ts)] ==== head-only 训练完成 ===="
echo "  best 产物: $O/train_params_best.npz(含 tok_scorer_*)"
echo "  下步: 修 infer.py 注入 tok_scorer → dyn 推理 + int8 + class_diag 对线 80.42"
echo "  对照: 起点 ens_r64_dyn int8=87.78/80.16 ; 线 87.91/80.42"
echo "[learnhead 完成 $(ts)]"
