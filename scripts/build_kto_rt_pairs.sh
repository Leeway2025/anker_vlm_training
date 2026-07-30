#!/bin/bash
# 定向 KTO 配对构建(RT 方向): 模型在训练集上自己犯的 RT 字母错 → 偏好对。
#   负样本 = 模型完整输出串(KTO v2 保险: 必须完整串),正样本 = GT 串。
# 用法: BASE 默认 seed-1;S5 上位后 BASE=outputs/jax_5b_s5/... 重跑本脚本。
#   nohup bash scripts/build_kto_rt_pairs.sh > kto_pairs.log 2>&1 &   (TPU ~2h)
# 产物: /data/kto_pairs_rt.jsonl;KTO 启动命令见文末注释。
set -e
cd "$(dirname "$0")/.."
BASE="${BASE:-outputs/jax_5b_seed1/train_params_best.npz}"
echo "[kto-rt] 底座 = $BASE"

# ① 训练集完整串推理(~2h)
mkdir -p outputs/kto_rt
bash jax_impl/infer_sharded.sh python /data/labels_dedup.jsonl /data/hf_layout.json \
    outputs/kto_rt/train_preds "$BASE" 8

# ② RT 字母错行 → 偏好对(CPU 秒级)
python3 - <<'PY'
import json, collections
gt = {}
for l in open('/data/labels_dedup.jsonl', encoding='utf-8'):
    d = json.loads(l); lb = d.get('labels') or d
    gt[d['video_id']] = (lb['role_type'], lb['sub_keyscene'],
                         str(lb.get('description', '')).strip())
conf = collections.Counter(); n_pairs = 0
with open('/data/kto_pairs_rt.jsonl', 'w', encoding='utf-8') as f:
    for l in open('outputs/kto_rt/train_preds.jsonl', encoding='utf-8'):
        d = json.loads(l)
        out = (d.get('output') or '').strip()
        seg = out.split('|')
        v = d['video_id']
        if v not in gt or len(seg) < 2:
            continue
        g_rt, g_sk, g_desc = gt[v]
        p_rt = seg[0].strip()
        if p_rt == g_rt or p_rt not in 'ABCDE':
            continue                       # 只收 RT 字母错的行
        conf[f'{g_rt}->{p_rt}'] += 1
        f.write(json.dumps({"video_id": v, "label": 1,
                            "completion": f"{g_rt}|{g_sk}|{g_desc}"},
                           ensure_ascii=False) + '\n')
        f.write(json.dumps({"video_id": v, "label": 0,
                            "completion": out}, ensure_ascii=False) + '\n')
        n_pairs += 1
print(f'[kto-rt] RT 错行 {n_pairs} 条(对数 x2 行)-> /data/kto_pairs_rt.jsonl')
print('[kto-rt] 方向分布:', conf.most_common(10))
print('[判读] A->D 应为最大头(测试侧 E 桶同款病);若训练集上 A->D 很少,'
      '说明模型在训练集上不犯这个错(记忆),KTO 此路收益存疑,先贴回结果再定')
PY

# ③ KTO 正式跑(冒烟过探针门槛后再放开;S5 排完的空槽执行):
# python3 jax_impl/kto.py --kto-data /data/kto_pairs_rt.jsonl \
#     --labels /data/labels_dedup.jsonl --layout /data/hf_layout.json \
#     --init-npz outputs/jax_5b_seed1/train_params_best.npz \
#     --kto-update b --lr 1e-6 --steps 400 --rank-scheme prod \
#     --probe-ids /data/val_ids_v2.txt --probe-every 50 \
#     --out outputs/kto_rt/run1
