#!/bin/bash
# 最大有用数据 anneal（用户 0811 05:20：加更多有用数据/两机并用/提准确率/暂不管压缩）
#   A 机：从 anneal_b_best 续 anneal 到 923k 全质量池（natural 分布），prod rank，seed 7。
#   与 B 机 seed2_base→923k seed11 互为汤原料；后续 soup(A,B,anneal_b,anneal_a)+公平校准。
#   断点续训 watchdog：抢占/崩溃自动从 ckpt_latest 爬起。
cd /workspace && unset WDS_DIR
O=outputs/maxdata_a
mkdir -p $O
ATT=0
until RESUME=""; [ -f $O/ckpt_latest.npz ] && RESUME="--resume"; \
  python jax_impl/train_sft.py $RESUME \
  --labels /data/labels_max_natural.jsonl \
  --layout /data/hf_layout.json \
  --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/anneal_b_best.npz \
  --augment --accum 32 \
  --lr 8e-6 --proj-lr 2e-4 --vision-lr 8e-6 \
  --warmup 100 --lr-schedule linear \
  --steps 2500 --eval-every 250 --early-stop-patience 6 \
  --ckpt-every 250 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out $O; do
  ATT=$((ATT+1)); echo "[retry] train exit, attempt $ATT $(date)"
  [ $ATT -ge 10 ] && exit 1
  sleep 60
done
[ -f $O/train_params_best.npz ] || { echo "[maxdata_a] 无产物"; exit 1; }

# 评测 + 公平校准
INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python \
  /data/labels_test.jsonl /data/hf_layout.json \
  $O/eval_preds $O/train_params_best.npz 8 && \
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt && \
python3 outputs/delivery_0807/fit_calibration.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --out $O/fitted_calibration.json > $O/calib.log 2>&1
echo "[maxdata_a] 完成 $(date)"
grep -i '重拟\|refit' $O/calib.log | tail -1
