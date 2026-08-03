#!/bin/bash
# 断点续训验收门(kill -9 中断 → resume → 与不中断对照逐点对齐)。
# 在 TPU 机上跑,约 20 分钟。过门标准(全自动判定):
#   ① B(被杀+续跑)与 A(一气跑完)的逐步 train loss / val_loss 一致;
#   ② 两者最终 train_params.npz 数值完全一致;
# 说明: 对照跑刻意不开 --augment —— 增强用共享 RNG,多线程下逐位复现
# 本就不成立(两次不中断的跑之间也不同);增强下的 resume 是"统计等价"
# (RNG 重播新的随机增强),正确性由本门禁的无增强逐位对齐背书。
# 用法: bash scripts/test_resume.sh   (环境变量 LABELS/LAYOUT 可覆盖)
set -e
cd "$(dirname "$0")/.."
unset WDS_DIR

LABELS=${LABELS:-/data/labels_dedup.jsonl}
LAYOUT=${LAYOUT:-/data/hf_layout.json}
A=outputs/resume_gate_a
B=outputs/resume_gate_b
rm -rf "$A" "$B"

COMMON="--labels $LABELS --layout $LAYOUT --val-n 32 --steps 24 --accum 4 \
  --eval-every 6 --seed 7 --prefetch-workers 8"

echo "== [1/3] 对照跑 A(不中断)=="
python jax_impl/train_sft.py $COMMON --out "$A"

echo "== [2/3] 实验跑 B: 跑到约 step 10 处 kill -9,再 resume 续完 =="
python jax_impl/train_sft.py $COMMON --ckpt-every 6 --resume --out "$B" &
PID=$!
# 轮询 metrics.jsonl,step≥10 即杀(落在 ckpt@6 与 ckpt@12 之间 →
# resume 会重放 7~10 步,顺带考验日志去重对账)
for i in $(seq 1 900); do
  sleep 2
  if ! kill -0 $PID 2>/dev/null; then
    echo "B 在被杀前就退出了(编译或数据问题),验收失败"; exit 1
  fi
  n=$(grep -c '"loss"' "$B/metrics.jsonl" 2>/dev/null || echo 0)
  [ "$n" -ge 10 ] && break
done
kill -9 $PID; wait $PID 2>/dev/null || true
echo "   已于 step≈$n 处 kill -9;60s 冷却(libtpu 锁)后 resume"
sleep 60
python jax_impl/train_sft.py $COMMON --ckpt-every 6 --resume --out "$B"

echo "== [3/3] 对账 =="
python3 - "$A" "$B" <<'PY'
import json
import sys

import numpy as np

A, B = sys.argv[1], sys.argv[2]

def load(d):
    tr, va = {}, {}
    for l in open(f"{d}/metrics.jsonl"):
        r = json.loads(l)
        if "loss" in r:
            tr[r["step"]] = r["loss"]        # 同 step 重复取最后(重放段)
        if "val_loss" in r:
            va[r["step"]] = r["val_loss"]
    return tr, va

ta, va_ = load(A)
tb, vb = load(B)
assert set(ta) == set(tb), f"step 集不齐: A-B={set(ta)-set(tb)} B-A={set(tb)-set(ta)}"
dt = max(abs(ta[s] - tb[s]) for s in ta)
dv = max(abs(va_[s] - vb[s]) for s in va_) if va_ else 0.0
print(f"train loss 最大差 {dt:.2e} | val_loss 最大差 {dv:.2e}(步数 {len(ta)})")
assert dt < 1e-5 and dv < 1e-5, "逐步 loss 不对齐 —— resume 状态有缺口"

za = np.load(f"{A}/train_params.npz")
zb = np.load(f"{B}/train_params.npz")
assert set(za.files) == set(zb.files), "参数叶集合不一致"
bad = [(k, float(np.abs(np.asarray(za[k], np.float32)
                        - np.asarray(zb[k], np.float32)).max()))
       for k in za.files
       if not np.array_equal(np.asarray(za[k]), np.asarray(zb[k]))]
if bad:
    bad.sort(key=lambda x: -x[1])
    raise AssertionError(f"最终参数 {len(bad)} 叶不一致,最大差 {bad[0]}")
print(f"最终参数 {len(za.files)} 叶逐位一致 ✓")
print("\n[验收通过] kill-resume 与不中断跑逐点等价,断点续训可上生产")
PY
