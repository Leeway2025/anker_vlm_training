#!/bin/bash
# 压缩域汤·剩余口径(0813):soup_tk32=sel2臂、chain_a7的soup_hyb2=hyb2s臂 均已覆盖,
# 这里只补唯一缺口 all4(select与hyb2两口径)。避免与 chain_a7 的 soup_hyb2 重复。
# 臂: tksel32b/_s2 (纯选 K=32) + tkhyb2b/_s2 (hyb2 K=30),同 init(soupw1)。
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/soup_all4

echo "[soup-rest] 构建 all4 权重汤(numpy, CPU)…"
python3 - <<'PY'
import numpy as np
P = {
 'sel':  'outputs/tksel32b/train_params_best.npz',
 'sels2':'outputs/tksel32b_s2/train_params_best.npz',
 'hyb':  'outputs/tkhyb2b/train_params_best.npz',
 'hybs2':'outputs/tkhyb2b_s2/train_params_best.npz',
}
D = {k: np.load(v) for k,v in P.items()}
ks = set(D['sel'].files)
for k in D: ks &= set(D[k].files)
def mix(weights):
    out={}
    for key in ks:
        if D['sel'][key].shape != D['sels2'][key].shape: continue
        s=None
        for name,w in weights.items():
            a=D[name][key].astype(np.float64)*w
            s=a if s is None else s+a
        out[key]=s.astype(np.float32)
    return out
np.savez('outputs/soup_all4/train_params_best.npz',  **mix({'sel':0.25,'sels2':0.25,'hyb':0.25,'hybs2':0.25}))
print('汤已建: soup_all4  键数=%d'%len(mix({'sel':1.0})))
PY

infer () { # $1=dir $2=mode(select|hyb2)
  local D=outputs/$1
  if [ "$2" = "hyb2" ]; then
    TOKEN_COMPRESS_MODE=hyb2 SELECT_TOKENS_K=30 INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
      bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
      $D/eval_preds $D/train_params_best.npz 8
  else
    SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits --rank-scheme prod" \
      bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
      $D/eval_preds $D/train_params_best.npz 8
  fi
  python3 jax_impl/eval_metrics.py --preds $D/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $D/eval_report.txt
}

echo "[soup-rest] ① all4 @ select K=32"
infer soup_all4 select
cp outputs/soup_all4/eval_preds.jsonl outputs/soup_all4/eval_preds_select.jsonl
cp outputs/soup_all4/eval_report.txt  outputs/soup_all4/eval_report_select.txt

echo "[soup-rest] ② all4 @ hyb2 K=30"
infer soup_all4 hyb2
cp outputs/soup_all4/eval_preds.jsonl outputs/soup_all4/eval_preds_hyb2.jsonl
cp outputs/soup_all4/eval_report.txt  outputs/soup_all4/eval_report_hyb2.txt

echo "[soup_k32_rest] 完成(仅 all4 两口径;hyb2s 由 chain_a7 的 soup_hyb2 覆盖) $(date)"
