#!/usr/bin/env bash
# 彻底停止 TPU 训练: 按"谁握着 TPU 设备"杀,而非按进程名匹配。
# 背景: torch_xla spawn 的子进程命令行是 `python -c from multiprocessing...`,
# 不含 train.py 字样,pkill -f train.py 杀不到它们 → /dev/vfio busy /
# 残留占 HBM(v6e-1 与客户 v6e-8 均实测踩坑)。
# 用法: bash scripts/stop_train.sh
set -e
found=0
for dev in /dev/vfio/[0-9]*; do
  [ -e "$dev" ] || continue
  pids=$(fuser "$dev" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true)
  for p in $pids; do
    echo "kill $p (holds $dev): $(ps -o cmd= -p "$p" | head -c 80)"
    kill -9 "$p" 2>/dev/null || true
    found=1
  done
done
[ "$found" = 0 ] && echo "没有进程占用 TPU 设备"
sleep 8
busy=0
for dev in /dev/vfio/[0-9]*; do
  [ -e "$dev" ] || continue
  fuser "$dev" >/dev/null 2>&1 && { echo "仍被占用: $dev"; busy=1; }
done
[ "$busy" = 0 ] && echo "✅ TPU 设备已全部释放,可以启动新训练"
