#!/bin/bash
# 压缩域 K=32 模型汤(0813):4 臂 best 权重平均 → 多口径推理 → 报告。
# 臂: tksel32b/_s2 (纯选 K=32) + tkhyb2b/_s2 (hyb2 merge K=30),同 init(soupw1)。
# 汤:
#   soup_sel2  = 0.5*(tksel32b + tksel32b_s2)         推理口径 select K=32
#   soup_hyb2s = 0.5*(tkhyb2b + tkhyb2b_s2)           推理口径 hyb2   K=30
#   soup_all4  = 0.25*四臂                             推理口径 select K=32 + hyb2 K=30 都测
# 跨方法权重汤=marquee(select×merge 不相关→红利最大);混合权重两种推理口径都评,取高。
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/soup_sel2 outputs/soup_hyb2s outputs/soup_all4

echo "[soup] 构建权重汤(numpy, CPU)…"
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
np.savez('outputs/soup_sel2/train_params_best.npz',  **mix({'sel':0.5,'sels2':0.5}))
np.savez('outputs/soup_hyb2s/train_params_best.npz', **mix({'hyb':0.5,'hybs2':0.5}))
np.savez('outputs/soup_all4/train_params_best.npz',  **mix({'sel':0.25,'sels2':0.25,'hyb':0.25,'hybs2':0.25}))
print('汤已建: soup_sel2 / soup_hyb2s / soup_all4  键数=%d'%len(mix({'sel':1.0})))
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

echo "[soup] ① all4 @ select K=32"
infer soup_all4 select
cp outputs/soup_all4/eval_preds.jsonl outputs/soup_all4/eval_preds_select.jsonl
cp outputs/soup_all4/eval_report.txt  outputs/soup_all4/eval_report_select.txt

echo "[soup] ② all4 @ hyb2 K=30"
infer soup_all4 hyb2
cp outputs/soup_all4/eval_preds.jsonl outputs/soup_all4/eval_preds_hyb2.jsonl
cp outputs/soup_all4/eval_report.txt  outputs/soup_all4/eval_report_hyb2.txt

echo "[soup] ③ sel2 @ select K=32"
infer soup_sel2 select

echo "[soup] ④ hyb2s @ hyb2 K=30"
infer soup_hyb2s hyb2

echo "[soup_k32_all] 完成 $(date)"
