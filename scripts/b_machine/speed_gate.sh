#!/bin/bash
# 吞吐包验收门(A 档): 生产配置 vs [bs2 + bf16 动量] 整包 A/B,自动裁决。
# 在 TPU 机上跑,约 1.5h(短程×2 + 64样本过拟合×2)。
#
# 包内容(作为一个单变量整体验收,过门后 train_1m.sh 改法见脚本尾输出):
#   --per-device-bs 2(批式视觉编码,accum 减半保持全局 batch 不变)
#   --mu-dtype bfloat16(代码默认,train_1m.sh 此前显式钉了 float32)
# 裁决标准:
#   ① 吞吐: samples/s 提升 ≥ 15%(不到不值得引入新路径);
#   ② 显存: peak < limit 的 95%(留碎片余量,防第 5000 步深夜 OOM);
#   ③ 平行: 同 seed 下两配置每个 opt step 消费同一批 64 样本(仅 micro 分组
#      不同),梯度数学同值 → 30 步后在同一份 val 卷上 val_loss 应仅差 bf16
#      求和序噪声(训练 loss 打印的是末 micro,两配置子集不同,不可比);
#   ④ 过拟合门禁(v7 铁律): 64 样本 150 步,吞吐包的塌陷终 loss 不得劣于
#      生产配置(比值 ≤1.5+0.02)—— 防"数值悄悄错但 loss 还在降"。
# 用法: bash scripts/speed_gate.sh  (LABELS/LAYOUT 环境变量可覆盖)
set -e
cd "$(dirname "$0")/.."
unset WDS_DIR

LABELS=${LABELS:-/data/labels_dedup.jsonl}
LAYOUT=${LAYOUT:-/data/hf_layout.json}
G=outputs/speed_gate
rm -rf "$G"; mkdir -p "$G"

# 64 样本固定子集(过拟合门禁用;seed 钉死保证可复现)
python3 - "$LABELS" "$G/labels_64.jsonl" <<'PY'
import random
import sys

rows = open(sys.argv[1], encoding="utf-8").readlines()
random.Random(0).shuffle(rows)
open(sys.argv[2], "w", encoding="utf-8").writelines(rows[:64])
PY

COMMON="--layout $LAYOUT --rank-scheme prod --train-vision --train-projector \
  --seed 7 --prefetch-workers 8"
# 生产配置(P)与吞吐包(T): 同全局 batch 64(8芯×bs×accum)。
# T 可用 SPEED_T_ARGS 覆盖以验其它组合(全局 batch 必须保持 64!),如:
#   SPEED_T_ARGS="--per-device-bs 2 --accum 4 --mu-dtype bfloat16 --remat-policy dots"
#   SPEED_T_ARGS="--per-device-bs 1 --accum 8 --mu-dtype bfloat16 --remat-policy dots"
P_ARGS="--per-device-bs 1 --accum 8 --mu-dtype float32"
T_ARGS="${SPEED_T_ARGS:---per-device-bs 2 --accum 4 --mu-dtype bfloat16}"
echo "[gate] 吞吐包配置: $T_ARGS"

SHORT="--steps 30 --eval-every 30 --val-n 64"   # step30 同卷 val = 轨迹对齐判据
echo "== [1/4] 短程 30 步 · 生产配置 =="
python jax_impl/train_sft.py $COMMON $P_ARGS --labels "$LABELS" \
  $SHORT --out "$G/short_p" 2>&1 | tee "$G/short_p.log" | tail -3
echo "== [2/4] 短程 30 步 · 吞吐包(此处 OOM 即判 bs2 不可行)=="
python jax_impl/train_sft.py $COMMON $T_ARGS --labels "$LABELS" \
  $SHORT --out "$G/short_t" 2>&1 | tee "$G/short_t.log" | tail -3
OV="--steps 150 --eval-every 0 --val-n 0"   # 关 eval: 64 样本一个不许漏进 val
echo "== [3/4] 64 样本过拟合 150 步 · 生产配置 =="
python jax_impl/train_sft.py $COMMON $P_ARGS --labels "$G/labels_64.jsonl" \
  $OV --out "$G/ov_p" 2>&1 | tee "$G/ov_p.log" | tail -2
echo "== [4/4] 64 样本过拟合 150 步 · 吞吐包 =="
python jax_impl/train_sft.py $COMMON $T_ARGS --labels "$G/labels_64.jsonl" \
  $OV --out "$G/ov_t" 2>&1 | tee "$G/ov_t.log" | tail -2

echo; echo "================ 裁决 ================"
python3 - "$G" <<'PY'
import json
import re
import sys

G = sys.argv[1]

def parse(log):
    sps, loss, hbm, val = [], {}, None, None
    for l in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"opt_step (\d+)/\d+ loss=([\d.]+) .*samples/s=([\d.]+)", l)
        if m:
            loss[int(m.group(1))] = float(m.group(2))
            sps.append(float(m.group(3)))
        m = re.search(r"peak=([\d.]+)G limit=([\d.]+)G", l)
        if m:
            hbm = (float(m.group(1)), float(m.group(2)))
        m = re.search(r"val_loss=([\d.]+)", l)
        if m:
            val = float(m.group(1))
    return sps, loss, hbm, val

sp, lp, hp, vp = parse(f"{G}/short_p.log")
st, lt, ht, vt = parse(f"{G}/short_t.log")
_, op, _, _ = parse(f"{G}/ov_p.log")
_, ot, _, _ = parse(f"{G}/ov_t.log")

warm = lambda v: sum(v[5:]) / max(len(v[5:]), 1)   # 跳过前5步(编译/预热)
v_p, v_t = warm(sp), warm(st)
uplift = (v_t / v_p - 1) * 100
dval = abs(vp - vt) if vp is not None and vt is not None else None
fin = lambda h: sum(h[s] for s in sorted(h)[-5:]) / 5
fp, ft = fin(op), fin(ot)

print(f"① 吞吐   生产 {v_p:.2f} → 吞吐包 {v_t:.2f} samples/s({uplift:+.1f}%)")
print(f"② 显存   吞吐包 peak {ht[0]:.2f}G / limit {ht[1]:.2f}G"
      f"({ht[0]/ht[1]:.0%})" if ht else "② 显存   未捕获 [hbm] 行")
print(f"③ 平行   step30 同卷 val_loss 生产 {vp} vs 吞吐包 {vt}"
      f"(Δ={dval if dval is None else round(dval, 4)})")
print(f"④ 过拟合 终 loss(末5步均)生产 {fp:.4f} vs 吞吐包 {ft:.4f}")

ok = [uplift >= 15, ht and ht[0] < 0.95 * ht[1],
      dval is not None and dval < 0.02,
      ft <= fp * 1.5 + 0.02]
tags = ["吞吐≥15%", "显存<95%", "val_loss Δ<0.02", "过拟合不劣"]
for t, o in zip(tags, ok):
    print(("  ✓ " if o else "  ✗ ") + t)
if all(ok):
    print("\n[放行] 吞吐包过门。train_1m.sh 改法(全局 batch 256 不变):")
    print("  --accum 32 → --accum 16;加 --per-device-bs 2;"
          "删 --mu-dtype float32(回到 bf16 默认)")
else:
    print("\n[拒绝] 未过门项见上。处置: 只挂 bf16 动量单项可另跑 "
          "P_ARGS 换 --mu-dtype bfloat16 的对照;bs2 显存超限则维持 bs1。")
    sys.exit(1)
PY
