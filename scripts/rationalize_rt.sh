#!/bin/bash
# RT 强化 CoT 重标注: 只重标 GT∈{A,D} 的行(A↔D=RT错误53%的主战场),
# 用 --rt-focus prompt(A/D 判据 + 懒标D严判),完成后与 v1 资产合并。
# 前置: export GEMINI_API_KEY=AQ.xxx ; export WDS_DIR=<训练集tar目录>
# 用法: nohup bash scripts/rationalize_rt.sh > rat_rt.log 2>&1 &
#   (Gemini 网络任务,可与 TPU 训练并行;断点续跑=原样重跑本脚本)
# 吞吐: A+D ~5万行,50并发 ≈ 2.5-3.5h
set -e
cd "$(dirname "$0")/.."
: "${WDS_DIR:?请先 export WDS_DIR=<WDS tar 目录>}"
# TPU 机自带的 GOOGLE_CLOUD_LOCATION=europe-west4 会把 Vertex 请求路由到
# 无 gemini-3.1-pro 的区域(404 NOT_FOUND 实测) —— 强制 global 端点
export GOOGLE_CLOUD_LOCATION=global
mkdir -p /data/assets_rat

# ① 圈定 A/D 行
python3 - <<'PY'
import json, collections
n = collections.Counter(); ids = []
for l in open('/data/labels_dedup.jsonl', encoding='utf-8'):
    d = json.loads(l); lb = d.get('labels') or d
    if lb['role_type'] in 'AD':
        ids.append(d['video_id']); n[lb['role_type']] += 1
open('/data/assets_rat/rt_focus_ids.txt', 'w').write('\n'.join(ids))
print(f"[scope] A={n['A']} D={n['D']} 合计 {len(ids)} 行待重标")
PY

# ② RT 强化重标(断点续跑安全: --out 里已成功的自动跳过)
python3 -m annotation.rationalize_cot \
    --wds-dir "$WDS_DIR" --labels /data/labels_dedup.jsonl \
    --only-ids /data/assets_rat/rt_focus_ids.txt --rt-focus \
    --out /data/assets_rat/rat_rt_raw.jsonl \
    --asset-out /data/assets_rat/asset_C_rt.jsonl \
    --workers 50

# ③ 合并: v1 资产剔掉 A/D 行,换上 RT 强化版(v2 判 unsupported 的 A/D 行
#    不再有链 = 懒标D的怯判教材整批消失)
python3 - <<'PY'
import json
gt = {}
for l in open('/data/labels_dedup.jsonl', encoding='utf-8'):
    d = json.loads(l); lb = d.get('labels') or d
    gt[d['video_id']] = lb['role_type']
keep = drop = 0
with open('/data/assets_rat/asset_C_rtfocus.jsonl', 'w', encoding='utf-8') as f:
    for l in open('/data/assets_rat/asset_C_reasoning.jsonl', encoding='utf-8'):
        v = json.loads(l)['video_id']
        if gt.get(v) in ('A', 'D'):
            drop += 1; continue
        f.write(l); keep += 1
    add = 0
    for l in open('/data/assets_rat/asset_C_rt.jsonl', encoding='utf-8'):
        f.write(l); add += 1
print(f'[merge] v1保留(B/C/E行) {keep} + RT强化 {add}(v1的A/D行弃 {drop})'
      f' -> asset_C_rtfocus.jsonl')
PY
echo "[OK] 训练时改挂 --cot-file /data/assets_rat/asset_C_rtfocus.jsonl 即为 S5-RT 版"
