#!/bin/bash
# B机夜班: 等 anneal_c 全链完 → 用 c 重配权重汤(两个候选)→ 直评+校准
# → 报告与优胜 npz 回传 GCS(B机系统盘用完即回收,产物必须离机)。
# 地板: 交付汤裸分 88.12/79.70 —— 候选不超它就继续交旧汤。
cd /workspace && unset WDS_DIR
ts() { date '+%m-%d %H:%M'; }

while [ ! -f outputs/jax_anneal_c/eval_report.txt ]; do sleep 300; done
sleep 30
echo "[$(ts)] anneal_c 出分:"; cat outputs/jax_anneal_c/eval_report.txt

C=outputs/jax_anneal_c/train_params_best.npz
B=outputs/anneal_b_best.npz
A=outputs/jax_anneal_a/train_params_best.npz
S=outputs/seed2_base/seed2_best.npz

mksoup() {  # mksoup <目录> <wc> <wb> <wa> <ws>
  local O=outputs/$1; mkdir -p $O
  [ -f $O/train_params_best.npz ] && return 0
  echo "[$(ts)] 配汤 $1: c=$2 b=$3 a=$4 s2=$5"
  WC=$2 WB=$3 WA=$4 WS=$5 OUT=$O python3 - <<'PY'
import numpy as np, os
w = [float(os.environ[k]) for k in ("WC", "WB", "WA", "WS")]
zs = [np.load(p) for p in (
    "outputs/jax_anneal_c/train_params_best.npz",
    "outputs/anneal_b_best.npz",
    "outputs/jax_anneal_a/train_params_best.npz",
    "outputs/seed2_base/seed2_best.npz")]
keys = set(zs[0].files)
for z in zs[1:]:
    keys &= set(z.files)
out = {}
for k in sorted(keys):
    if not all(z[k].shape == zs[0][k].shape for z in zs[1:]):
        continue
    acc = sum(wi * z[k].astype(np.float64) for wi, z in zip(w, zs) if wi)
    out[k] = acc.astype(np.float32)
np.savez(os.environ["OUT"] + "/train_params_best.npz", **out)
print(f'{len(out)} 键, 权重 c/b/a/s2 = {w}')
PY
}

evalone() {  # evalone <目录>
  local O=outputs/$1
  [ -f $O/eval_report.txt ] && return 0
  echo "[$(ts)] === $1 推理评测 ==="
  INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
    /data/labels_test.jsonl /data/hf_layout.json \
    $O/eval_preds $O/train_params_best.npz 8 || return 1
  python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $O/eval_report.txt
  [ -f scripts/apply_calibration.py ] && \
    python3 scripts/apply_calibration.py $O/eval_preds.jsonl \
      --gold /data/labels_test.jsonl | tee $O/calibrated_report.txt
}

mksoup soup_c622  0.6 0.0 0.2 0.2 && evalone soup_c622   # b→c 直换
mksoup soup_c4321 0.4 0.3 0.2 0.1 && evalone soup_c4321  # c/b 同锅

echo "[$(ts)] B机夜班汇总(裸口径):"
for d in jax_anneal_c soup_c622 soup_c4321; do
  echo "== $d"; grep -E 'RoleType|SubKS' outputs/$d/eval_report.txt 2>/dev/null
done
# 产物离机(B盘会回收): 报告全传, npz 传所有新品(明早在A机汇总裁决)
gcloud storage cp outputs/jax_anneal_c/eval_report.txt \
  outputs/soup_c*/eval_report.txt outputs/soup_c*/calibrated_report.txt \
  outputs/soup_c*/eval_preds.jsonl outputs/jax_anneal_c/eval_preds.jsonl \
  gs://zx_vlm_dataset/backup_bmachine_0810/ 2>&1 | tail -1
for d in soup_c622 soup_c4321; do
  gcloud storage cp outputs/$d/train_params_best.npz \
    gs://zx_vlm_dataset/backup_bmachine_0810/$d.npz 2>&1 | tail -1
done
echo "[$(ts)] B机夜班完成"
