#!/bin/bash
# A机主链v4(0814晚,神谕已判选择饱和→撤显著性步骤):
#   ①受害集K48反事实 → ②60k错误标注(阶段②原料)→ ③重采样器阶段①热身
cd /workspace
busy() { ls /proc/*/cmdline 2>/dev/null | xargs -I{} sh -c "tr \"\0\" \" \" < {} 2>/dev/null" 2>/dev/null | grep -v chain_a15 | grep -qE "train_sft|infer\.py|extract_saliency|resampler_warmup"; }
wait_idle() { while true; do ok=1; for t in 1 2 3; do busy && { ok=0; break; }; sleep 60; done; [ $ok -eq 1 ] && return; sleep 120; done; }
step_retry() { local name=$1; shift; for t in 1 2 3 4 5 6; do wait_idle; "$@" && return 0; echo "[a15] $name 重试$t $(date)" >> outputs/chain_a15.log; sleep 240; done; return 1; }
s1() { SELECT_TOKENS_K=48 INFER_ARGS="--rank-scheme prod" bash jax_impl/infer_sharded.sh python outputs/victims_labels.jsonl /data/hf_layout.json outputs/victims_k48/eval_preds outputs/soup_tk32/train_params_best.npz 8 > outputs/victims_k48.log 2>&1 && python3 jax_impl/eval_metrics.py --preds outputs/victims_k48/eval_preds.jsonl --labels outputs/victims_labels.jsonl > outputs/victims_k48/eval_report.txt 2>&1 && grep -q RoleType outputs/victims_k48/eval_report.txt; }
s2() { SELECT_TOKENS_K=32 INFER_ARGS="" bash jax_impl/infer_sharded.sh python outputs/train60k.jsonl /data/hf_layout.json outputs/train60k_pred/eval_preds outputs/r64_ens_kd/train_params_best.npz 8 > outputs/train60k_pred.log 2>&1 && [ -f outputs/train60k_pred/eval_preds.jsonl ]; }
s3() { bash scripts/resampler_warmup.sh > outputs/rsp_warmup.log 2>&1 && ls outputs/rsp_warmup*/train_params_best.npz >/dev/null 2>&1 || bash scripts/resampler_warmup.sh > outputs/rsp_warmup.log 2>&1; }
step_retry victims s1 && echo "[a15] ①victims done $(date)" >> outputs/chain_a15.log
step_retry errlabel s2 && echo "[a15] ②60k标注 done $(date)" >> outputs/chain_a15.log
step_retry rspwarm s3 && echo "[a15] ③重采样热身 done $(date)" >> outputs/chain_a15.log
echo "[a15] ALL DONE $(date)" >> outputs/chain_a15.log
