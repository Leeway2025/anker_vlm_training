#!/bin/bash
# 0816 r64终局(用户授权:客户自有验收集,本地test=内部代理,80%并入训练):
# 等种子汤链跑完 → init=汤(若汤评测更好)否则r64v2 → train+test80 KD 400步
# → int8 → dyn 推理【全量test出preds】+ 洁净20% held-out 出报告
cd /workspace
while [ ! -f outputs/r64_soup3.done ]; do sleep 300; done
sleep 60
# 选init:比较汤 vs r64v2 的int8+dyn报告(裸SubKS)
INIT=outputs/r64v2_ens/train_params_best.npz
SOUP_SK=$(grep SubKS outputs/r64_soup3/eval_report.txt 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | head -1)
V2_SK=78.33
if [ -n "$SOUP_SK" ] && python3 -c "exit(0 if float($SOUP_SK)>float($V2_SK) else 1)"; then
  INIT=outputs/r64_soup3/soup3.npz
fi
echo "[t80] init=$INIT soup_sk=$SOUP_SK" >> outputs/r64_t80.log
export SELECT_TOKENS_K=32
python jax_impl/train_sft.py   --labels outputs/labels_train_plus_test80.jsonl   --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt   --rank-scheme uniform --rank 64   --init-npz $INIT   --teacher-npz outputs/soup_size/teacher_u512.npz   --teacher-npz2 outputs/soup_size/teacher2_hyb2b_u512.npz   --distill-coef 0.5 --distill-temp 2.0   --train-vision --train-projector   --sample-weights outputs/sw_rare_test80.json   --augment --accum 16   --lr 5e-6 --proj-lr 1e-5 --vision-lr 5e-6 --loraplus-ratio 1   --warmup 15 --lr-schedule linear --seed 3 --rt-w 8   --steps 400 --eval-every 50 --out outputs/r64_t80 > outputs/r64_t80_train.log 2>&1
python3 outputs/delivery_0807/quantize_lora.py --in outputs/r64_t80/train_params_best.npz   --out-bf16 /dev/null --out-int8 outputs/r64_t80/t80_int8.npz > outputs/r64_t80/quant.log 2>&1
TOKEN_COMPRESS_MODE=dyn SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits"   bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json   outputs/r64_t80/eval_int8dyn outputs/r64_t80/t80_int8.npz 8 > outputs/r64_t80_infer.log 2>&1
python3 jax_impl/eval_metrics.py --preds outputs/r64_t80/eval_int8dyn.jsonl   --labels outputs/test_heldout.jsonl > outputs/r64_t80/eval_report_heldout.txt 2>&1
echo done >> outputs/r64_t80.done
