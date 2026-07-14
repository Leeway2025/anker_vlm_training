#!/usr/bin/env bash
# 训练启动器: 自带 TPU 环境变量,不依赖 ~/.bashrc(agent 沙箱/非交互
# shell/nohup/systemd 下 bashrc 不会加载,曾致客户现场变量"设而无效")。
# 用法(参数原样透传给 train.py):
#   bash scripts/run_train.sh --phase configs/phase5_sft.yaml --stage b \
#       --init-from outputs/phase5_sft_a/final
set -e
cd "$(dirname "$0")/.."

export PJRT_DEVICE=TPU
# 非标准 metadata 机器(GKE/自定义镜像)必需;标准 Cloud TPU VM 无害
export TPU_SKIP_MDS_QUERY=1
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-8}"   # 按机型覆盖
export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,4,1}"
# 切勿在此设 TPU_PROCESS_BOUNDS/TPU_CHIPS_PER_PROCESS_BOUNDS(libtpu 会
# 误读为多进程分片声明 → INVALID_ARGUMENT;torch_xla spawn 时自行注入)

echo "[run_train] TPU env:"; env | grep -E "^(TPU_|PJRT_)" | sed 's/^/  /'
exec python3 training/train.py "$@"
