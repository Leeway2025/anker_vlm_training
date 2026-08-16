#!/bin/bash
# 诊断:学习打分头影响力增益扫描(不重训,推理时读 env TOKEN_LEARN_GAIN)。
# 目的=回应"逻辑不对/代码问题吧":若加大 gain 能大幅改变选择/输出→接线正确、
# gain=8 只是影响力弱(结论欠力);若 gain=200 仍几乎不改输出→接线/逻辑有 bug。
set -e
cd /workspace
P=outputs/learnhead_r64/train_params_best.npz
mkdir -p outputs/gain_sweep
for G in 0 8 50 200; do
  echo "=== GAIN=$G ==="
  TPU_VISIBLE_CHIPS=0 TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_ADDRESSES=localhost:8500 TPU_PROCESS_PORT=8500 CLOUD_TPU_TASK_ID=0 \
  TOKEN_LEARN_SCORE=1 TOKEN_LEARN_GAIN=$G SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn \
  python jax_impl/infer.py --labels /data/labels_test.jsonl --layout /data/hf_layout.json \
    --limit 200 --init-npz $P --out outputs/gain_sweep/g$G.jsonl > outputs/gain_sweep/g$G.log 2>&1
  echo "done G=$G lines=$(wc -l < outputs/gain_sweep/g$G.jsonl)"
done
echo "=== diff outputs vs gain=0 ==="
python3 -c "
import json
def load(f):
    d={}
    for l in open(f):
        j=json.loads(l); d[j['id']]=j.get('output','')
    return d
base=load('outputs/gain_sweep/g0.jsonl')
for G in [8,50,200]:
    cur=load(f'outputs/gain_sweep/g{G}.jsonl')
    diff=sum(1 for k in base if base[k]!=cur.get(k))
    ldiff=sum(1 for k in base if base[k][:1]!=cur.get(k,'')[:1])
    sdiff=sum(1 for k in base if base[k][:3]!=cur.get(k,'')[:3])
    print(f'gain={G}: 全文不同 {diff}/{len(base)}; RoleType字母不同 {ldiff}; 前3字符(RT|SubKS)不同 {sdiff}')
"
