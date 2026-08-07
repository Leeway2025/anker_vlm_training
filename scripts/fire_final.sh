#!/bin/bash
# 终版点火总编排(deadline 周五16:00服): 等半池打分+冒烟 → 批次口音过滤
# → 重建700k(不足则降档) → 分发 → 双机双seed点火。全自动, 决策记录到日志。
set -e
LOG=/home/nas-tpu-poc/fire_final.log
ts() { date '+%m-%d %H:%M'; }
say() { echo "[$(ts)] $*" >> $LOG; }
KEY=/home/nas-tpu-poc/.ssh/key_tpu_new
B=nas-tpu-poc@10.164.0.11
say "总编排启动"

# ① 等 B 机后半池打分(scores.jsonl 合并文件出现)
while ! ssh -i $KEY $B 'test -f ~/code/anker_vlm_training/outputs/gate2_full_b/scores.jsonl'; do sleep 300; done
scp -i $KEY -q $B:~/code/anker_vlm_training/outputs/gate2_full_b/scores.jsonl /data/gate2_half_b_scores.jsonl
say "半池分数到手 $(wc -l < /data/gate2_half_b_scores.jsonl) 行"

# ② 等冒烟 eval
while [ ! -f /mnt/disks/data/anker_vlm_training/outputs/jax_p5_smoke/eval_report.txt ]; do sleep 300; done
SMOKE=$(grep "SubKS" /mnt/disks/data/anker_vlm_training/outputs/jax_p5_smoke/eval_report.txt | head -1)
say "冒烟判决: $SMOKE (基线73.52±0.5)"

# ③ 批次口音地图 + 重建池(python)
python3 /mnt/disks/data/anker_vlm_training/scripts/build_pool_final.py >> $LOG 2>&1

# ④ 吞吐终校: 用冒烟末100步实测步速定档(700k 若赶不上 deadline-8h 则降 500k)
SPS=$(sudo docker exec tpu_train bash -c "grep -oE 'samples/s=[0-9.]+' /workspace/outputs/jax_p5_smoke/train.log | tail -100 | cut -d= -f2 | awk '{s+=\$1} END {print s/NR}'")
say "实测吞吐 ${SPS} samples/s"
PICK=$(python3 -c "
sps=float('${SPS}' or 12)
import time
left=(time.mktime(time.strptime('2026-08-07 16:00','%Y-%m-%d %H:%M'))-time.time())/3600
need7=700000*2/256*(256/sps)/3600+3
print('700k' if left-8>need7 else '500k')")
say "定档: $PICK(余量规则 deadline-8h)"
LB=/data/pool_${PICK}_final_labels.jsonl
STEPS=$([ "$PICK" = 700k ] && echo 5469 || echo 3906)

scp -i $KEY -q $LB $B:~/data/
say "池已分发, steps=$STEPS"

# ⑤ 点火: A=seed1(排锁), B=seed2
for SEED in 1 2; do
  OUT=outputs/jax_final_s$SEED
  CMD="cd /workspace && unset WDS_DIR && ATT=0; until python jax_impl/train_sft.py --labels $LB --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 4 --accum 32 --steps $STEPS --eval-every 250 --val-n 1657 --seed $SEED --mu-dtype float32 --prefetch-workers 24 --cartography --ckpt-every 250 --resume --out $OUT; do ATT=\$((ATT+1)); [ \$ATT -ge 8 ] && exit 1; sleep 60; done; INFER_ARGS='--dump-letter-logits' bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json $OUT/eval_preds $OUT/train_params_best.npz 8 && python3 jax_impl/eval_metrics.py --preds $OUT/eval_preds.jsonl --labels /data/labels_test.jsonl | tee $OUT/eval_report.txt"
  if [ $SEED = 1 ]; then
    sudo docker exec -d tpu_train bash -c "exec 200>/tmp/night_chain.lock; flock -w 43200 200; nohup bash -c \"$CMD\" > /workspace/logs/final_s1.log 2>&1 &"
    say "A机 seed1 已排队(等冒烟链释放锁)"
  else
    ssh -i $KEY $B "sudo docker exec -d tpu_train bash -c 'nohup bash -c \"$CMD\" > /workspace/logs/final_s2.log 2>&1 &'"
    say "B机 seed2 已点火"
  fi
done
say "总编排完成 —— 双seed在途, 各自训完自动 eval+dump logits"
