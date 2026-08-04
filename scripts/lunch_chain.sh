#!/bin/bash
# 午间小任务链(08-04): 全部无人值守、幂等、与夜链同锁串行。
#   ① test_resume.sh   断点续训验收门(~20min, 顺带踩 per-sample loss 新路径)
#   ② cartography 冒烟  6 步小跑验证 --cartography 落盘格式
#   ③ speed_gate.sh    吞吐包验收门(~1.5h, bs2+bf16动量 整包 A/B)
# 日志: logs/lunch_chain.<ts>.log
set -e
cd "$(dirname "$0")/.."
unset WDS_DIR
exec 200>/tmp/night_chain.lock
flock -n 200 || { echo "[FATAL] 有链持锁, 退出"; exit 1; }
ts() { date '+%m-%d %H:%M:%S'; }

echo "[$(ts)] ① 断点续训验收门 test_resume.sh"
bash scripts/test_resume.sh

echo "[$(ts)] ② cartography 冒烟(6 步)"
rm -rf outputs/carto_smoke
python jax_impl/train_sft.py --labels /data/labels_dedup.jsonl \
  --layout /data/hf_layout.json --val-n 32 --steps 6 --accum 4 \
  --eval-every 0 --seed 7 --prefetch-workers 8 --cartography \
  --out outputs/carto_smoke
n=$(wc -l < outputs/carto_smoke/cartography.jsonl)
head -2 outputs/carto_smoke/cartography.jsonl
[ "$n" -ge 100 ] || { echo "[FATAL] cartography 行数 $n 异常(6步×32样本应≈192)"; exit 1; }
echo "[$(ts)] cartography 冒烟通过: $n 行"

echo "[$(ts)] ③ 吞吐包验收门 speed_gate.sh"
bash scripts/speed_gate.sh

echo "[$(ts)] 午间链全部完成"
