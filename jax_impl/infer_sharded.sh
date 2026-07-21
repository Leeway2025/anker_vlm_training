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

pids=()
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
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" -ne 0 ]; then
  echo "❌ 有分片失败 —— 保留全部分片文件(断点续跑: 原命令重跑只补缺失)"
  echo "   排错看 ${OUT}_shard*.log 尾部"
  exit 1
fi
cat "${OUT}"_shard*.jsonl > "${OUT}.jsonl"
n=$(wc -l < "${OUT}.jsonl")
echo "[jax_infer_sharded] $n preds -> ${OUT}.jsonl"
empty=$(grep -c '"output": ""' "${OUT}.jsonl" 2>/dev/null || true)
if [ "${empty:-0}" -gt 0 ]; then
  echo "⚠️ ${empty}/${n} 条空输出 —— 比例高则勿直接评测(见 infer 日志哨兵)"
fi
# 全分片成功才清中间文件(失败路径保留 = 断点续跑凭据);KEEP_SHARDS=1 可保留
if [ -z "${KEEP_SHARDS:-}" ]; then
  cat "${OUT}"_shard*.log "${OUT}"_shard*.jsonl.log > "${OUT}.infer.log" 2>/dev/null || true
  rm -f "${OUT}"_shard*.jsonl "${OUT}"_shard*.log "${OUT}"_shard*.jsonl.log
  echo "[jax_infer_sharded] 中间文件已清理(日志并入 ${OUT}.infer.log;KEEP_SHARDS=1 可保留)"
fi
