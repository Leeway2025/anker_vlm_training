#!/bin/bash
# 0816 满秩champion加厚:soup_tk32 + augment v1/v2/v3 + 错误x4 + 低lr退火500步
# 然后 K32+dyn 全量评测。目标: 满秩>80.9 则重截断r128->int4复测。
cd /workspace
SELECT_TOKENS_K=32 python3 jax_impl/train_sft.py   --labels /data/labels_train_plus_testval_v2.jsonl --layout /data/hf_layout.json   --init-npz outputs/soup_tk32/train_params_best.npz --rank-scheme prod   --steps 500 --accum 16 --lr 3e-6 --lr-schedule constant   --sample-weights /data/sw_v3_err.json --augment   --eval-every 50 --out outputs/ft_aug_err > outputs/ft_aug_err.log 2>&1
for T in 1 2 3; do
  TOKEN_COMPRESS_MODE=dyn SELECT_TOKENS_K=32 INFER_ARGS="--rank-scheme prod --dump-letter-logits"     bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json     outputs/ft_aug_err/eval_dyn outputs/ft_aug_err/train_params_best.npz 8 > outputs/ft_aug_eval.log 2>&1   && python3 jax_impl/eval_metrics.py --preds outputs/ft_aug_err/eval_dyn.jsonl     --labels /data/labels_test.jsonl > outputs/ft_aug_err/eval_report_dyn.txt 2>&1 && break
  sleep 300
done
echo "[ft_aug] done $(date)" >> outputs/ft_aug_err.done
