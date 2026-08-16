#!/bin/bash
# vfio-gated 接力:等 dynseg(chain_probes)腾卡后,依次补跑两条被 13:10 抢卡撞崩的探针:
#   ① r96rec_dyn(交付判据:dyn输入+r96恢复权重,双压缩;shards 0-3,5 撞 vfio busy 需续)
#   ② tome(零样本换轴探针;部分分片失败需续)
# 各自:infer_sharded 续跑(只补缺失分片)→ 官方裸评 → class_diag 公平校准。
# 纪律:每棒开跑前先等 vfio 真空(comm 过滤 python/python3 + /proc/<pid>/fd 查 vfio),不抢卡。
cd /workspace && unset WDS_DIR
ts() { date '+%m-%d %H:%M'; }

wait_vfio() {
  echo "[chain-r96dyn $(ts)] 等 vfio 释放 …"
  while :; do
    hold=0
    for p in $(ls /proc | grep -E '^[0-9]+$'); do
      comm=$(cat /proc/$p/comm 2>/dev/null)
      case "$comm" in python|python3) : ;; *) continue;; esac
      if ls -l /proc/$p/fd 2>/dev/null | grep -q vfio; then hold=1; break; fi
    done
    [ "$hold" = 0 ] && break
    sleep 20
  done
  echo "[chain-r96dyn $(ts)] vfio 空"
}

run_calib() {  # $1=输出目录标签
  local O=$1
  python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
    --labels /data/labels_test.jsonl | tee $O/eval_report.txt
  python3 outputs/class_diag.py $O/eval_preds.jsonl \
    --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
    2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt
}

# ===== ① r96rec_dyn(交付判据,优先)=====
wait_vfio
echo "[chain-r96dyn $(ts)] 续跑 r96rec_dyn(dyn输入+r96恢复权重,补缺失分片)"
SELECT_TOKENS_K=32 MAX_SOFT_TOKENS=64 TOKEN_COMPRESS_MODE=dyn INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  outputs/r96rec_dyn/eval_preds outputs/r96_recover/train_params_best.npz 8
echo "[chain-r96dyn $(ts)] r96rec_dyn 评测+校准"
run_calib outputs/r96rec_dyn
echo "[chain-r96dyn $(ts)] ==== r96rec_dyn 汇总 ===="
echo "  校准: $(cat outputs/r96rec_dyn/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  参照: r96恢复(均匀)=87.88/80.42(RT差0.03) ; soup_tk32+dyn=88.21/80.87 ; 线 87.91/80.42"

# ===== ② tome(零样本换轴探针)=====
wait_vfio
echo "[chain-r96dyn $(ts)] 续跑 tome(补缺失分片)"
TOKEN_COMPRESS_MODE=tome TOME_TOTAL=512 SELECT_TOKENS_K=0 MAX_SOFT_TOKENS=64 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  outputs/probe_tome/eval_preds outputs/soupw1/soupw1.npz 8
echo "[chain-r96dyn $(ts)] tome 评测+校准"
run_calib outputs/probe_tome
echo "[chain-r96dyn $(ts)] ==== tome 汇总 ===="
echo "  校准: $(cat outputs/probe_tome/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  参照: soupw1@16x64=88.37/81.17 ; 8x64零样本=79.28 ; 16x32选择零样本=77.24 ; 线 87.91/80.42"
echo "[chain-r96dyn 完成 $(ts)]"
