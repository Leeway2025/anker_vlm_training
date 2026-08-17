#!/bin/bash
# KTO 续训一条龙(~2.5h): 配对扩挖(RT或SK字母错,CPU) → KTO 400步(探针自门禁,
#   不超基线不落盘) → 评测+先验 → 交付候选(对 73.77/83.65)
# 用法: nohup bash scripts/kto_run.sh > kto_run.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
# 防环境污染: rationalize 用的 WDS_DIR(训练集目录)若被继承,会把测试集
# 推理导向错误 tar(KeyError 实测)。链内数据路径由 labels/meta 自解析。
unset WDS_DIR
test -s outputs/kto_rt/train_preds.jsonl || { echo "[kto] 缺训练集完整串,先跑 build_kto_rt_pairs.sh"; exit 1; }

# ① 配对扩挖: RT 或 SK 任一字母错 → 偏好对(CPU 秒级,零推理)
python3 - <<'PY'
import json, collections
gt = {}
for l in open('/data/labels_dedup.jsonl', encoding='utf-8'):
    d = json.loads(l); lb = d.get('labels') or d
    gt[d['video_id']] = (lb['role_type'], lb['sub_keyscene'],
                         str(lb.get('description', '')).strip())
n = collections.Counter()
with open('/data/kto_pairs_all.jsonl', 'w', encoding='utf-8') as f:
    for l in open('outputs/kto_rt/train_preds.jsonl', encoding='utf-8'):
        d = json.loads(l); out = (d.get('output') or '').strip()
        seg = out.split('|'); v = d['video_id']
        if v not in gt or len(seg) < 3: continue
        g_rt, g_sk, g_desc = gt[v]
        rt_bad = seg[0].strip() != g_rt
        sk_bad = seg[1].strip() != g_sk
        if not (rt_bad or sk_bad): continue
        n['rt_err'] += rt_bad; n['sk_err'] += sk_bad; n['pairs'] += 1
        f.write(json.dumps({"video_id": v, "label": 1,
                            "completion": f"{g_rt}|{g_sk}|{g_desc}"},
                           ensure_ascii=False) + '\n')
        f.write(json.dumps({"video_id": v, "label": 0,
                            "completion": out}, ensure_ascii=False) + '\n')
print(f"[kto] 配对 {n['pairs']} 条(RT错 {n['rt_err']} / SK错 {n['sk_err']})"
      f" -> /data/kto_pairs_all.jsonl")
PY

# ② KTO 续训(五件保险: ref=起点/仅B矩阵/lr 1e-6/探针step-1基线/超基线才落盘)
python3 jax_impl/kto.py --kto-data /data/kto_pairs_all.jsonl \
    --labels /data/labels_dedup.jsonl --layout /data/hf_layout.json \
    --init-npz outputs/jax_5b_seed1/train_params_best.npz \
    --kto-update b --lr 1e-6 --steps 400 --rank-scheme prod \
    --probe-ids /data/val_ids_v2.txt --probe-every 50 \
    --out outputs/kto_run1
if [ ! -s outputs/kto_run1/lora_params_best.npz ]; then
  echo "[kto] 全程未超基线,零伤害出局(交付口径不变 73.77),收工"
  exit 0
fi

# ③ 评测 + 先验 → 交付候选
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/kto_run1/eval_preds outputs/kto_run1/lora_params_best.npz 8
python3 jax_impl/eval_metrics.py --preds outputs/kto_run1/eval_preds.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/kto_run1/eval_report.txt
mkdir -p outputs/optin_kto
INFER_ARGS='--dump-letter-logits' \
bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
    outputs/optin_kto/preds outputs/kto_run1/lora_params_best.npz 8
python3 jax_impl/apply_surgical_prior.py --logits outputs/optin_kto/preds.jsonl \
    --labels /data/labels_test.jsonl \
    --fold-a /data/test_sfoldA.jsonl --fold-b /data/test_sfoldB.jsonl \
    --tau 0.7 --out outputs/optin_kto/preds_surg.jsonl
python3 jax_impl/eval_metrics.py --preds outputs/optin_kto/preds_surg.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/optin_kto/eval_report.txt
echo "[kto] 全链完成 $(date) —— optin_kto/eval_report.txt 对 73.77/83.65"
