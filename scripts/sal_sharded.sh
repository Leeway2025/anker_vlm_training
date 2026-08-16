#!/usr/bin/env bash
# 老师梯度显著性 8 芯并行抽取 + 合并。
#   bash scripts/sal_sharded.sh <init_npz> <rank_scheme> <out_prefix> [limit]
# 产物: <out_prefix>.npz(sg,gi) + <out_prefix>.ids.json
set -e
cd "$(dirname "$0")/.."
NPZ=${1:?init_npz}
SCHEME=${2:?rank_scheme}
OUT=${3:?out_prefix}
LIMIT=${4:-0}
N=8
LIM_ARG=""; [ "$LIMIT" != "0" ] && LIM_ARG="--limit $LIMIT"
mkdir -p "$(dirname "$OUT")"
echo "[sal_sharded] $N 芯抽取 saliency init=$NPZ scheme=$SCHEME → $OUT"
pids=()
for i in $(seq 0 $((N - 1))); do
  TPU_VISIBLE_CHIPS=$i TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_ADDRESSES=localhost:$((8476 + i)) TPU_PROCESS_PORT=$((8476 + i)) \
  CLOUD_TPU_TASK_ID=0 \
  python3 jax_impl/attrib_saliency.py --labels /data/labels_test.jsonl \
    --layout /data/hf_layout.json --init-npz "$NPZ" --rank-scheme "$SCHEME" \
    --shard "$i/$N" $LIM_ARG --out "${OUT}_shard${i}" \
    > "${OUT}_shard${i}.launch.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -ne 0 ] && { echo "❌ 有分片失败,看 ${OUT}_shard*.launch.log"; exit 1; }
python3 - "$OUT" "$N" <<'PY'
import sys, json, numpy as np
out, n = sys.argv[1], int(sys.argv[2])
sg, gi, ids = [], [], []
for i in range(n):
    z = np.load(f"{out}_shard{i}.npz")
    sg.append(z["sg"]); gi.append(z["gi"])
    ids += json.load(open(f"{out}_shard{i}.ids.json"))
np.savez(out + ".npz", sg=np.concatenate(sg), gi=np.concatenate(gi))
json.dump(ids, open(out + ".ids.json", "w"))
print(f"[sal_sharded] 合并 M={len(ids)} → {out}.npz")
PY
rm -f "${OUT}"_shard*.npz "${OUT}"_shard*.ids.json "${OUT}"_shard*.launch.log
echo "[sal_sharded] 完成"
