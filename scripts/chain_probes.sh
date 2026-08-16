#!/bin/bash
# 压缩换轴零样本对照:tome(时序ToMe合并)→ dynseg(动态分段预算),各独立跑(一个失败不挡另一个)。
# 均 soupw1 零样本、16帧源、总预算512、公平校准 vs 线 87.91/80.42。
# 首次上设备:tome/dynseg 仅过 CPU 单测,device 端首跑,失败看各自 O/eval_preds_shard*.log。
cd /workspace
ts() { date '+%m-%d %H:%M'; }
echo "[chain-probes $(ts)] 起 tome 零样本"
bash scripts/probe_tome.sh 2>&1 | tee outputs/probe_tome.log || echo "[chain-probes $(ts)] tome FAILED(见 outputs/probe_tome/eval_preds_shard*.log)"
echo "[chain-probes $(ts)] 起 dynseg 零样本"
bash scripts/probe_dynseg.sh 2>&1 | tee outputs/probe_dynseg.log || echo "[chain-probes $(ts)] dynseg FAILED(见 outputs/probe_dynseg/eval_preds_shard*.log)"
echo "[chain-probes $(ts)] ===== 对照汇总 ====="
echo "  tome  校准: $(cat outputs/probe_tome/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  dynseg校准: $(cat outputs/probe_dynseg/fair_calib.txt 2>/dev/null | tr '\n' ' ')"
echo "  对照 soupw1@16x64=88.37/81.17 ; 8x64零样本=79.28 ; 16x32选择零样本=77.24 ; 线 87.91/80.42"
echo "[chain-probes 完成 $(ts)]"
