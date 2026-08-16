#!/bin/bash
# 后处理接力:等已在跑的 vr_k32_zero 分片推理(infer_sharded, PPid 990081)产出合并 eval_preds.jsonl,
# 再跑官方裸评 + class_diag 公平校准(与线 87.91/80.42 同口径)。配置=topk K32/soft64(交付部署口径)。
# 回答网页指令 "vr上k32":vr_distill_attn 在 K=32 token 压缩下是否仍过线。无 set -e;vfio 门控不抢卡。
cd /workspace && unset WDS_DIR
ts() { date '+%m-%d %H:%M'; }
O=outputs/vr_k32_zero

echo "[vr-k32-calib $(ts)] 等分片推理产出 eval_preds.jsonl …"
# 等合并文件出现且分片进程退出(infer_sharded 结束时合并)
while :; do
  running=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    comm=$(cat /proc/$p/comm 2>/dev/null) || true
    case "$comm" in python|python3) : ;; *) continue;; esac
    if ls -l /proc/$p/fd 2>/dev/null | grep -q vfio; then
      # 确认是本任务:cmdline 含 vr_k32_zero
      cl=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
      case "$cl" in *vr_k32_zero*) running=1; break;; esac
    fi
  done
  [ "$running" = 0 ] && [ -s "$O/eval_preds.jsonl" ] && break
  # 分片进程已退但合并文件未出:可能还在合并,再等
  [ "$running" = 0 ] && [ ! -s "$O/eval_preds.jsonl" ] && { echo "[vr-k32-calib $(ts)] 推理进程已退,等合并/或需补分片…"; sleep 10; ls -s "$O/eval_preds.jsonl" 2>/dev/null; }
  sleep 15
done

echo "[vr-k32-calib $(ts)] eval_preds.jsonl 就绪($(wc -l < $O/eval_preds.jsonl) 行);官方裸评"
python3 jax_impl/eval_metrics.py --preds $O/eval_preds.jsonl \
  --labels /data/labels_test.jsonl | tee $O/eval_report.txt
echo "[vr-k32-calib $(ts)] class_diag 公平校准(同线口径)"
python3 outputs/class_diag.py $O/eval_preds.jsonl \
  --gold /data/labels_test.jsonl --train /data/labels_train_plus_testval_v2.jsonl \
  2>&1 | grep -iE "n=11022|RT_cal|SK_cal" | tee $O/fair_calib.txt

echo "[vr-k32-calib $(ts)] ==== vr_distill_attn + K=32(topk/soft64) 汇总 ===="
echo "  校准: $(cat $O/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照: vr 全token(无压缩)=88.00/80.46 ; 线 87.91/80.42 ; r64+dyn=87.56/80.12"
echo "  判据: K32 下仍过线→交付有效(部署口径); 掉线→需在 K32 下补训 或 换 dyn 口径"
echo "[vr-k32-calib 完成 $(ts)]"
