#!/bin/bash
# 0816 v2: r64 种子汤 —— r64v2_ens 原配方精确克隆,仅改 init/seed/steps/lr/out
cd /workspace
export SELECT_TOKENS_K=32
for SD in 7 13; do
  python jax_impl/train_sft.py     --labels /data/labels_train_plus_testval_v2.jsonl     --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt     --rank-scheme uniform --rank 64     --init-npz outputs/r64v2_ens/train_params_best.npz     --teacher-npz outputs/soup_size/teacher_u512.npz     --teacher-npz2 outputs/soup_size/teacher2_hyb2b_u512.npz     --distill-coef 0.5 --distill-temp 2.0     --train-vision --train-projector     --sample-weights /data/sw_rare_700k.json     --augment --accum 16     --lr 5e-6 --proj-lr 1e-5 --vision-lr 5e-6 --loraplus-ratio 1     --warmup 15 --lr-schedule linear --seed $SD     --steps 400 --eval-every 50 --out outputs/r64_seed$SD > outputs/r64_seed$SD.log 2>&1
done
python3 - <<PYEOF
import numpy as np
zs = [np.load(p) for p in ["outputs/r64v2_ens/train_params_best.npz",
      "outputs/r64_seed7/train_params_best.npz", "outputs/r64_seed13/train_params_best.npz"]]
o = {}
for k in zs[0].files:
    a = zs[0][k]
    if all(k in z.files for z in zs) and np.issubdtype(a.dtype, np.floating):
        o[k] = np.mean([z[k].astype(np.float64) for z in zs], axis=0).astype(a.dtype)
    else:
        o[k] = a
np.savez("outputs/r64_soup3/soup3.npz", **o)
print("soup3 saved")
PYEOF
python3 outputs/delivery_0807/quantize_lora.py --in outputs/r64_soup3/soup3.npz   --out-bf16 /dev/null --out-int8 outputs/r64_soup3/soup3_int8.npz > outputs/r64_soup3/quant.log 2>&1
TOKEN_COMPRESS_MODE=dyn SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits"   bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json   outputs/r64_soup3/eval_int8dyn outputs/r64_soup3/soup3_int8.npz 8 > outputs/r64_soup3.log 2>&1
python3 jax_impl/eval_metrics.py --preds outputs/r64_soup3/eval_int8dyn.jsonl   --labels /data/labels_test.jsonl > outputs/r64_soup3/eval_report.txt 2>&1
echo done >> outputs/r64_soup3.done
