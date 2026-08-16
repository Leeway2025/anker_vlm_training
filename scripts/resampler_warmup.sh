#!/bin/bash
# ========================= 阶段① 重采样器热身(草稿,勿自动 arm)=========================
# 目标: TOKEN_RESAMPLER=1 的可学习重采样器(512 query cross-attn 1024 soft token)
#       先回归『现有 dyn K=32 选择的 512 token』(MSE,teacher 同批现算,无外部标签),
#       只训 tok_resampler_*(~11M 参),视觉编码器冻结、LoRA/LLM 完全不参与。
# 产物: outputs/resampler_warmup/resampler_warmup.npz(键 tok/<路径>)
# 阶段②(后续,另开脚本): train_sft TOKEN_RESAMPLER=1 --init-npz 本产物 + 现有底座
#       npz 合并(或直接 --init-npz 底座产物后手动并入 tok/ 键)端到端适配。
# ⚠️ 运维红线(0813 教训): 起卡前必查 vfio 占用 + COORDINATION 认领,勿与在跑链互杀;
#    本脚本仅为草稿,须人工确认 TPU 空闲后手动执行。
# CPU 冒烟(不占 TPU,先验证跑通):
#   docker exec tpu_train sh -c 'cd /workspace && JAX_PLATFORMS=cpu TOKEN_RESAMPLER=1 \
#     python3 jax_impl/resampler_warmup.py --labels /data/labels_100k_v2.jsonl \
#     --limit 24 --val-n 8 --bs 2 --steps 3 --eval-every 2 --out outputs/rsp_smoke'
set -e
cd /workspace
O=outputs/resampler_warmup
mkdir -p $O
ts() { date '+%m-%d %H:%M'; }

# ---- vfio 真空等待(与 ens_r64_dyn.sh 同款;TPU 忙则一直等)----
echo "[rsp_warmup $(ts)] 等 vfio 真空 …"
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

echo "[rsp_warmup $(ts)] 起阶段① 热身(dyn K=32 teacher,MSE,只训重采样器)"
# 口径说明:
#   TOKEN_RESAMPLER=1        —— 与阶段②/推理同一开关(热身循环本身不走补丁,
#                               仅保证 RESAMPLER_* 读数一致)
#   RESAMPLER_TOKENS=512     —— 与 dyn K=32(16 帧 × 32 = 512)严格同预算
#   --k 32                   —— teacher = _compress_soft_tokens(t,32,"dyn"),
#                               与生产 dyn 口径(floor8+全局竞争、保光栅序)一致
#   数据用 100k 池即可(回归目标是几何/内容对齐,不需要全量 700k)
TOKEN_RESAMPLER=1 RESAMPLER_TOKENS=512 RESAMPLER_LAYERS=1 RESAMPLER_HEADS=8 \
MAX_SOFT_TOKENS=64 \
python jax_impl/resampler_warmup.py \
  --labels /data/labels_100k_v2.jsonl \
  --steps 2000 --bs 8 --lr 1e-4 --k 32 --seed 0 \
  --val-n 64 --eval-every 100 \
  --out $O

echo "[rsp_warmup $(ts)] 完成 → $O/resampler_warmup.npz + warmup_meta.json"
echo "[rsp_warmup] 阶段② 预计改动点(已核对,均为既有通道,无新机制):"
echo "  1) 底座热启: 把本 npz 的 tok/ 键并入所选底座产物 npz(python 一行 savez 合并),"
echo "     train_sft TOKEN_RESAMPLER=1 --init-npz <合并产物> —— restore_train_tree"
echo "     自动命中 tok/ 叶(train_sft 已支持,零代码改动);"
echo "  2) lr 分组: 重采样器走 'tok' 组(--tok-lr,建议 1e-4 起,LoRA/proj 照旧);"
echo "  3) 推理: infer.py TOKEN_RESAMPLER=1 --init-npz <阶段②产物>(已接注入通道);"
echo "  4) 蒸馏暂不可用(老师前向缺 tok_resampler_* 参数,train_sft 已显式拦截)。"
