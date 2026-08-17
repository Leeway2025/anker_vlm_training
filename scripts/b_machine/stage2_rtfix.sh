#!/bin/bash
# 二阶段续训模板(RT 修复第一发,~2.5h): seed-1 起步 + RT位加权x8 + 身份词加权x3
#   lr=峰值1/10,400步,val v2 门槛;评测+先验全表对 73.77/83.65,不涨即回退
# 用法: nohup bash scripts/stage2_rtfix.sh > stage2_rtfix.log 2>&1 &
# BASE 可换(如 seed-2/汤上位后): BASE=outputs/xxx/train_params_best.npz bash ...
set -e
cd "$(dirname "$0")/.."
unset WDS_DIR
BASE="${BASE:-outputs/jax_5b_seed1/train_params_best.npz}"
echo "[st2] 续训起点 = $BASE | $(date) | $(git log --oneline -1)"

python jax_impl/train_sft.py --labels /data/labels_dedup.jsonl \
    --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
    --init-npz "$BASE" --augment --early-stop-patience 3 \
    --accum 32 --steps 400 --lr 2e-6 --eval-every 50 --val-ids /data/val_ids_v2.txt \
    --seed 1 --mu-dtype float32 --rt-w 8 --idw 3 \
    --prefetch-workers 24 --out outputs/jax_5b_st2rt
echo "[st2] 训完 $(date)"

bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/jax_5b_st2rt/eval_preds outputs/jax_5b_st2rt/train_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/jax_5b_st2rt/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/jax_5b_st2rt/eval_report.txt
mkdir -p outputs/optin_st2rt
INFER_ARGS='--dump-letter-logits' \
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/optin_st2rt/preds outputs/jax_5b_st2rt/train_params_best.npz 8
python3 jax_impl/apply_surgical_prior.py --logits outputs/optin_st2rt/preds.jsonl \
    --labels /data/labels_test.jsonl \
    --fold-a /data/test_sfoldA.jsonl --fold-b /data/test_sfoldB.jsonl \
    --tau 0.7 --out outputs/optin_st2rt/preds_surg.jsonl
python3 jax_impl/eval_metrics.py --preds outputs/optin_st2rt/preds_surg.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/optin_st2rt/eval_report.txt
echo "[st2] 全链完成 —— optin_st2rt/eval_report.txt 对 73.77/83.65(RT>=+0.3 且 SubKS回撤<=0.2 才上位)"
