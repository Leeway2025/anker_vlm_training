#!/bin/bash
# ===== 阶段② 学习式重采样器端到端适配(input token 减半 = 1024→512 软聚合)=====
# 目标: 用重采样器(软聚合, 不硬丢弃)替代 dyn K=32 硬选择, 看能否越过 hard-K32
#       在 122MB 档撞到的 SubKS≈80.16 天花板。
# 底座: ens_r64_dyn/train_params_best.npz(dyn K=32 双师KD 适配的 r64, 80.16)
#       已并入 resampler_warmup 的 tok/ 键 → train_params_best_rsp.npz。
# 注: TOKEN_RESAMPLER=1 与蒸馏互斥(老师前向缺 tok_resampler_*), 故本阶段纯 CE;
#     KD 知识已烤进 ens_r64_dyn 底座权重, 此处仅 CE 适配重采样输入分布。
set -e
cd /workspace
O=outputs/rsp_stage2
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }

# ---- vfio 真空等待(与既有链同款)----
echo "[rsp2 $(ts)] 等 vfio 真空 …"
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

echo "[rsp2 $(ts)] 起阶段② 端到端适配(resampler 512 软聚合, 纯CE, tok-lr 1e-4)"
TOKEN_RESAMPLER=1 RESAMPLER_TOKENS=512 RESAMPLER_LAYERS=1 RESAMPLER_HEADS=8 \
RESAMPLER_FFN=1 RESAMPLER_FFN_MULT=2 MAX_SOFT_TOKENS=64 \
python jax_impl/train_sft.py \
  --labels /data/labels_train_plus_testval_v2.jsonl --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 \
  --init-npz outputs/ens_r64_dyn/train_params_best_rsp.npz \
  --train-vision --train-projector \
  --sample-weights /data/sw_rare_700k.json --augment \
  --accum 16 --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 --tok-lr 1e-4 \
  --loraplus-ratio 1 --warmup 30 --lr-schedule linear \
  --steps 600 --eval-every 50 --early-stop-patience 4 --ckpt-every 200 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O

echo "[rsp2 $(ts)] 训练完成 → $O/train_params_best.npz ; 下步: 推理 TOKEN_RESAMPLER=1 + class_diag"
