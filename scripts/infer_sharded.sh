#!/usr/bin/env bash
# 多芯并行推理: 每芯一个独立进程,按 --shard i/n 切数据,末尾合并。
#
# 背景(客户现场): 单进程 run_inference 只有一张芯在算(其余 7 张
# 只是被 PJRT 挂了进程没活干),10 万级训练集推理要 10+ 小时。
# 本脚本用 TPU_VISIBLE_CHIPS 把每个进程限到单芯 → 全芯并行。
#
# 用法(分片数缺省 = 自动检测本机全部 TPU 芯):
#   bash scripts/infer_sharded.sh outputs/phase5_sft_b/final \
#        DATA/labels.jsonl [outputs/preds] [分片数]
# 产物: outputs/preds.jsonl;分片文件保留(outputs/preds_shard*.jsonl),
# 各分片独立断点续跑 —— 中断后重跑本脚本只补缺失部分。
set -e
cd "$(dirname "$0")/.."
CKPT=${1:?checkpoint 目录}
LABELS=${2:?labels.jsonl}
OUT=${3:-outputs/preds}

detect_chips() {
  local n
  n=$(ls /dev/accel* 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then echo "$n"; return; fi
  n=$(ls /dev/vfio 2>/dev/null | grep -cE '^[0-9]+$')
  if [ "$n" -gt 0 ]; then echo "$n"; return; fi
  echo 8   # v6e-8 兜底
}
N=${4:-$(detect_chips)}
echo "[infer_sharded] 使用 $N 张芯并行(ckpt=$CKPT)"

for i in $(seq 0 $((N - 1))); do
  PJRT_DEVICE=TPU TPU_VISIBLE_CHIPS=$i \
  python3 training/run_inference.py --ckpt "$CKPT" --labels "$LABELS" \
    --shard "$i/$N" --out "${OUT}_shard${i}.jsonl" \
    > "${OUT}_shard${i}.log" 2>&1 &
done
wait
cat "${OUT}"_shard*.jsonl > "${OUT}.jsonl"
echo "[infer_sharded] $(wc -l < "${OUT}.jsonl") preds -> ${OUT}.jsonl"
