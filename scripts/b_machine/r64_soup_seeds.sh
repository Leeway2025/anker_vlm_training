#!/bin/bash
# 0816 r64-int8 最后一招: 同基底种子汤
# 从 r64v2(act-stats基底)出 2 个种子变体(400步短KD, seed 7/13),与原版三方权重平均
# → int8+dyn 评测。基底同源=汤合法(0813 跨SVD基底教训不适用)。
cd /workspace
for SD in 7 13; do
  SELECT_TOKENS_K=32 TOKEN_COMPRESS_MODE=dyn python3 jax_impl/train_sft.py     --labels /data/labels_train_plus_testval_v2.jsonl --layout /data/hf_layout.json     --init-npz outputs/r64v2_ens/train_params_best.npz --rank-scheme uniform     --teacher-npz outputs/soup_size/teacher_u512.npz     --teacher-npz2 outputs/soup_size/teacher2_hyb2b_u512.npz     --steps 400 --accum 16 --lr 5e-6 --seed $SD     --distill-coef 0.5 --distill-temp 2.0     --sample-weights /data/sw_rare_700k.json     --eval-every 50 --out outputs/r64_seed$SD > outputs/r64_seed$SD.log 2>&1
done
python3 - <<PYEOF
import numpy as np
zs = [np.load(p) for p in ["outputs/r64v2_ens/train_params_best.npz",
      "outputs/r64_seed7/train_params_best.npz", "outputs/r64_seed13/train_params_best.npz"]]
keys = zs[0].files
o = {}
for k in keys:
    if any(k not in z.files for z in zs):
        o[k] = zs[0][k]; continue
    a = zs[0][k]
    if np.issubdtype(a.dtype, np.floating):
        o[k] = np.mean([z[k].astype(np.float64) for z in zs], axis=0).astype(a.dtype)
    else:
        o[k] = a
np.savez("outputs/r64_soup3/soup3.npz", **o)
print("soup3 saved", len(keys))
PYEOF
python3 outputs/delivery_0807/quantize_lora.py --in outputs/r64_soup3/soup3.npz   --out-bf16 /dev/null --out-int8 outputs/r64_soup3/soup3_int8.npz > outputs/r64_soup3/quant.log 2>&1
TOKEN_COMPRESS_MODE=dyn SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits"   bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json   outputs/r64_soup3/eval_int8dyn outputs/r64_soup3/soup3_int8.npz 8 > outputs/r64_soup3.log 2>&1
python3 jax_impl/eval_metrics.py --preds outputs/r64_soup3/eval_int8dyn.jsonl   --labels /data/labels_test.jsonl > outputs/r64_soup3/eval_report.txt 2>&1
echo "[r64_soup3] done $(date)" >> outputs/r64_soup3.done
