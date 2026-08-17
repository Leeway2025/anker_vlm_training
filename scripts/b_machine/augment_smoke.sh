#!/bin/bash
# 增强 v2 端到端冒烟(08-04 午间): --augment --augment-v2 30 步真跑,
# 验证 ①不崩 ②loss 正常下降 ③吞吐损耗可接受(v2 是 CPU 侧增强,
# 预取 24 workers 下应被掩蔽)。效果消融(S2b2)在今晚干净数据上做,
# 本脚本只验机械可用性。阻塞等 night_chain 锁, 排在午间链之后。
set -e
cd "$(dirname "$0")/.."
unset WDS_DIR
exec 200>/tmp/night_chain.lock
flock -w 14400 200 || { echo "[FATAL] 等锁 4h 超时"; exit 1; }
ts() { date '+%m-%d %H:%M:%S'; }

echo "[$(ts)] 增强 v2 冒烟开跑(30 步, v1+v2 全开)"
rm -rf outputs/augv2_smoke
python jax_impl/train_sft.py --labels /data/labels_dedup.jsonl \
  --layout /data/hf_layout.json --rank-scheme prod --train-vision \
  --train-projector --augment --augment-v2 --steps 30 --accum 4 \
  --val-n 32 --eval-every 0 --seed 7 --prefetch-workers 24 \
  --out outputs/augv2_smoke
echo "[$(ts)] 冒烟跑完, 自检:"
python3 - <<'PY'
import json
ls = [json.loads(l)["loss"] for l in open("outputs/augv2_smoke/metrics.jsonl")
      if "loss" in l and "val" not in l]
assert len(ls) >= 30, f"步数不足 {len(ls)}"
head, tail = sum(ls[:5])/5, sum(ls[-5:])/5
print(f"loss 首5步均 {head:.3f} -> 末5步均 {tail:.3f}")
assert tail < head, "30 步 loss 未下降 —— v2 增强疑似破坏输入"
print("[PASS] 增强 v2 机械冒烟通过(不崩+loss 下降)")
PY
grep -oE "samples/s=[0-9.]+" outputs/augv2_smoke/train.log | tail -3
echo "[$(ts)] 完成。效果消融今晚做(干净数据 v2 开/关 单变量)"
