#!/bin/bash
# 0816 零接触rt-w8: t80的held-out显示rt-w8独立抬RT到87.9裸分
# → 去掉980条混入,纯训练集+rt-w8复跑,若过线=干净的r64-int8过线方案
cd /workspace
export SELECT_TOKENS_K=32
python jax_impl/train_sft.py   --labels /data/labels_train_plus_testval_v2.jsonl   --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt   --rank-scheme uniform --rank 64   --init-npz outputs/r64v2_ens/train_params_best.npz   --teacher-npz outputs/soup_size/teacher_u512.npz   --teacher-npz2 outputs/soup_size/teacher2_hyb2b_u512.npz   --distill-coef 0.5 --distill-temp 2.0   --train-vision --train-projector   --sample-weights /data/sw_rare_700k.json   --augment --accum 16   --lr 5e-6 --proj-lr 1e-5 --vision-lr 5e-6 --loraplus-ratio 1   --warmup 15 --lr-schedule linear --seed 3 --rt-w 8   --steps 400 --eval-every 50 --out outputs/r64_rtw > outputs/r64_rtw_train.log 2>&1
python3 outputs/delivery_0807/quantize_lora.py --in outputs/r64_rtw/train_params_best.npz   --out-bf16 /dev/null --out-int8 outputs/r64_rtw/rtw_int8.npz > outputs/r64_rtw/quant.log 2>&1
TOKEN_COMPRESS_MODE=dyn SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits"   bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json   outputs/r64_rtw/eval_int8dyn outputs/r64_rtw/rtw_int8.npz 8 > outputs/r64_rtw_infer.log 2>&1
python3 jax_impl/eval_metrics.py --preds outputs/r64_rtw/eval_int8dyn.jsonl   --labels /data/labels_test.jsonl > outputs/r64_rtw/eval_report.txt 2>&1
echo done >> outputs/r64_rtw.done
