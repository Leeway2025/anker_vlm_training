#!/usr/bin/env bash
# JAX 多芯并行推理: 每芯一个独立进程 × --shard i/n 分片,末尾合并。
# 与 torch 版 scripts/infer_sharded.sh 同构;各分片独立断点续跑
# (中断后重跑本脚本只补缺失部分)。
#
# 用法(分片数缺省 = 自动检测本机全部 TPU 芯):
#   bash jax_impl/infer_sharded.sh <venv_jax>/bin/python \
#        DATA/labels.jsonl hf_layout.json outputs/preds \
#        [lora.npz] [分片数]
set -e
cd "$(dirname "$0")/.."
PY=${1:?venv_jax 的 python 路径}
LABELS=${2:?labels.jsonl}
LAYOUT=${3:?hf_layout.json}
OUT=${4:-outputs/preds}
NPZ=${5:-}
detect_chips() {
  local n
  n=$(ls /dev/accel* 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then echo "$n"; return; fi
  n=$(ls /dev/vfio 2>/dev/null | grep -cE '^[0-9]+$')
  if [ "$n" -gt 0 ]; then echo "$n"; return; fi
  echo 8
}
N=${6:-$(detect_chips)}
NPZ_ARG=""
[ -n "$NPZ" ] && NPZ_ARG="--init-npz $NPZ"
echo "[jax_infer_sharded] 使用 $N 张芯并行"

for i in $(seq 0 $((N - 1))); do
  # 并发单芯进程隔离(实测): 仅 TPU_VISIBLE_CHIPS 会撞 libtpu 锁
  # ("TPU already in use"),须补进程边界与独立端口
  TPU_VISIBLE_CHIPS=$i \
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_ADDRESSES=localhost:$((8476 + i)) \
  TPU_PROCESS_PORT=$((8476 + i)) CLOUD_TPU_TASK_ID=0 \
  "$PY" jax_impl/infer.py --labels "$LABELS" --layout "$LAYOUT" \
    --shard "$i/$N" --out "${OUT}_shard${i}.jsonl" $NPZ_ARG \
    > "${OUT}_shard${i}.log" 2>&1 &
done
wait
cat "${OUT}"_shard*.jsonl > "${OUT}.jsonl"
echo "[jax_infer_sharded] $(wc -l < "${OUT}.jsonl") preds -> ${OUT}.jsonl"
