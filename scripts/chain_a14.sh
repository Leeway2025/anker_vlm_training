#!/bin/bash
# A机主链v3(0814):三连空闲确认 + 每步带重试,防TOCTOU烧穿
cd /workspace
busy() { ls /proc/*/cmdline 2>/dev/null | xargs -I{} sh -c "tr \"\0\" \" \" < {} 2>/dev/null" 2>/dev/null | grep -v chain_a14 | grep -qE "train_sft|infer\.py|extract_saliency"; }
wait_idle() { while true; do ok=1; for t in 1 2 3; do busy && { ok=0; break; }; sleep 60; done; [ $ok -eq 1 ] && return; sleep 120; done; }
step_retry() { local name=$1; shift; for t in 1 2 3 4 5 6; do wait_idle; "$@" && return 0; echo "[a14] $name 失败重试$t $(date)" >> outputs/chain_a14.log; sleep 240; done; return 1; }

s1() { SELECT_TOKENS_K=48 INFER_ARGS="--rank-scheme prod" bash jax_impl/infer_sharded.sh python outputs/victims_labels.jsonl /data/hf_layout.json outputs/victims_k48/eval_preds outputs/soup_tk32/train_params_best.npz 8 > outputs/victims_k48.log 2>&1 && python3 jax_impl/eval_metrics.py --preds outputs/victims_k48/eval_preds.jsonl --labels outputs/victims_labels.jsonl > outputs/victims_k48/eval_report.txt 2>&1; }
s2() { local fail=0; for i in 0 1 2 3 4 5 6 7; do TPU_VISIBLE_CHIPS=$i TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_ADDRESSES=localhost:$((8576+i)) TPU_PROCESS_PORT=$((8576+i)) CLOUD_TPU_TASK_ID=0 python3 jax_impl/extract_saliency.py --init-npz outputs/soupw1/soupw1.npz --labels outputs/test2k.jsonl --layout /data/hf_layout.json --n 2000 --shard $i/8 --out outputs/saliency/test2k_shard$i.npz > outputs/saliency/t2k_$i.log 2>&1 & done; wait; for i in 0 1 2 3 4 5 6 7; do [ -f outputs/saliency/test2k_shard$i.npz ] || fail=1; done; [ $fail -eq 0 ]; }
s3() { SELECT_TOKENS_K=32 INFER_ARGS="" bash jax_impl/infer_sharded.sh python outputs/train60k.jsonl /data/hf_layout.json outputs/train60k_pred/eval_preds outputs/r64_ens_kd/train_params_best.npz 8 > outputs/train60k_pred.log 2>&1 && [ -f outputs/train60k_pred/eval_preds.jsonl ]; }
s4() { local fail=0; for i in 0 1 2 3 4 5 6 7; do TPU_VISIBLE_CHIPS=$i TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_ADDRESSES=localhost:$((8576+i)) TPU_PROCESS_PORT=$((8576+i)) CLOUD_TPU_TASK_ID=0 python3 jax_impl/extract_saliency.py --init-npz outputs/soupw1/soupw1.npz --labels /data/labels_train_plus_testval_v2.jsonl --layout /data/hf_layout.json --n 50000 --shard $i/8 --out outputs/saliency/sal_shard$i.npz > outputs/saliency/shard$i.log 2>&1 & done; wait; for i in 0 1 2 3 4 5 6 7; do [ -f outputs/saliency/sal_shard$i.npz ] || fail=1; done; [ $fail -eq 0 ]; }

step_retry victims s1 && echo "[a14] ①victims done $(date)" >> outputs/chain_a14.log
step_retry s0maps s2 && echo "[a14] ②S0图 done $(date)" >> outputs/chain_a14.log
step_retry errlabel s3 && echo "[a14] ③60k标注 done $(date)" >> outputs/chain_a14.log
step_retry trainsal s4 && echo "[a14] ④训练显著性 done $(date)" >> outputs/chain_a14.log
echo "[a14] ALL DONE $(date)" >> outputs/chain_a14.log
