#!/bin/bash
# v2(08-05): B机提早点火——池构建只需半池口音地图; A机仍等冒烟(占着锁)。
set -e
LOG=/home/nas-tpu-poc/fire_final.log
ts() { date '+%m-%d %H:%M'; }
say() { echo "[$(ts)] $*" >> $LOG; }
KEY=/home/nas-tpu-poc/.ssh/key_tpu_new
B=nas-tpu-poc@10.164.0.11
say "编排v2启动(B提早点火版)"

while ! ssh -i $KEY $B 'test -f ~/code/anker_vlm_training/outputs/gate2_full_b/scores.jsonl'; do sleep 180; done
scp -i $KEY -q $B:~/code/anker_vlm_training/outputs/gate2_full_b/scores.jsonl /data/gate2_half_b_scores.jsonl
say "半池分数 $(wc -l < /data/gate2_half_b_scores.jsonl) 行"
python3 /mnt/disks/data/anker_vlm_training/scripts/build_pool_final.py >> $LOG 2>&1
LB=/data/pool_700k_final_labels.jsonl; STEPS=5469
say "定档700k(实测15/s, 25h内完, 余量充足)"
scp -i $KEY -q $LB $B:~/data/

CMD_TPL='cd /workspace && unset WDS_DIR && ATT=0; until python jax_impl/train_sft.py --labels LBF --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 4 --accum 32 --steps NSTEP --eval-every 250 --val-n 1657 --seed NSEED --mu-dtype float32 --prefetch-workers 24 --cartography --ckpt-every 250 --resume --out OUTD; do ATT=$((ATT+1)); [ $ATT -ge 8 ] && exit 1; sleep 60; done; INFER_ARGS="--dump-letter-logits" bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json OUTD/eval_preds OUTD/train_params_best.npz 8 && python3 jax_impl/eval_metrics.py --preds OUTD/eval_preds.jsonl --labels /data/labels_test.jsonl | tee OUTD/eval_report.txt'

# B: 立即点火 seed2
C2=$(echo "$CMD_TPL" | sed "s|LBF|$LB|g; s|NSTEP|$STEPS|g; s|NSEED|2|g; s|OUTD|outputs/jax_final_s2|g")
ssh -i $KEY $B "sudo docker exec -d tpu_train bash -c 'nohup bash -c \"$C2\" > /workspace/logs/final_s2.log 2>&1 &'"
say "B机 seed2 已提早点火"

# A: 等冒烟出分(判决记录用) → 抢占器放锁 → seed1 排锁
while [ ! -f /mnt/disks/data/anker_vlm_training/outputs/jax_p5_smoke/eval_report.txt ]; do sleep 180; done
say "冒烟判决: $(grep 'SubKS' /mnt/disks/data/anker_vlm_training/outputs/jax_p5_smoke/eval_report.txt | head -1)(基线73.52±0.5)"
C1=$(echo "$CMD_TPL" | sed "s|LBF|$LB|g; s|NSTEP|$STEPS|g; s|NSEED|1|g; s|OUTD|outputs/jax_final_s1|g")
sudo docker exec -d tpu_train bash -c "exec 200>/tmp/night_chain.lock; flock -w 43200 200; nohup bash -c \"$C1\" > /workspace/logs/final_s1.log 2>&1 &"
say "A机 seed1 已排锁点火"
say "编排v2完成"
