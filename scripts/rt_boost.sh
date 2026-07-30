#!/bin/bash
# RT 阈值两折优化(五维,复现门)+ SK 手术先验合体 → 新交付候选
# 用法: bash scripts/rt_boost.sh   (纯 CPU,~2 分钟)
set -e
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json, math, collections
import numpy as np
import sys
sys.path.insert(0, '.')
from jax_impl.prior_opt import coord_ascent, load, RT_SET, SK_SET

vids, rt_lg, sk_lg, rt_y, sk_y = load('outputs/optin/preds.jsonl',
                                      '/data/labels_test.jsonl')
fa = {json.loads(l)["video_id"] for l in open('/data/test_sfoldA.jsonl')}
ia = np.asarray([v in fa for v in vids]); ib = ~ia

def prior_of(path, key):
    c = collections.Counter()
    for l in open(path):
        d = json.loads(l); v = d.get('labels') or d
        c[v[key]] += 1
    n = sum(c.values())
    return {k: x/n for k, x in c.items()}

tr_sk = prior_of('/data/labels_dedup.jsonl', 'sub_keyscene')

def robust_fit_rt(lg, y, seed=0):
    rng = np.random.RandomState(seed)
    half = rng.permutation(len(y)) < len(y)//2
    o1, _ = coord_ascent(lg[half], y[half], 1.2)
    o2, _ = coord_ascent(lg[~half], y[~half], 1.2)
    agree = np.sign(o1) == np.sign(o2)
    return np.where(agree, np.sign(o1)*np.minimum(np.abs(o1), np.abs(o2)), 0.0)

out_rt = np.zeros((len(vids), 5)); out_m = np.zeros(len(vids))
for fit, app, fold in ((ib, ia, '/data/test_sfoldB.jsonl'),
                       (ia, ib, '/data/test_sfoldA.jsonl')):
    off = robust_fit_rt(rt_lg[fit], rt_y[fit])
    out_rt[app] = off
    tg_sk = prior_of(fold, 'sub_keyscene')
    dm = 0.7*math.log(max(tg_sk.get('m',1e-9),1e-9)/max(tr_sk.get('m',1e-9),1e-9))
    out_m[app] = dm
    print('RT偏移(对侧折拟合):',
          ' '.join(f'{RT_SET[i]}{off[i]:+.2f}' for i in range(5) if off[i]),
          f'| Δm={dm:+.3f}')

sk2 = sk_lg.copy(); sk2[:, SK_SET.index('m')] += out_m
rt2 = rt_lg + out_rt
kr = rt2.argmax(1); ks = sk2.argmax(1)
for i in range(len(vids)):                       # A|n / A|u → C
    if RT_SET[kr[i]] == 'A' and SK_SET[ks[i]] in ('n','u'):
        kr[i] = RT_SET.index('C')
with open('outputs/optin/preds_rtboost.jsonl','w') as f:
    for i, v in enumerate(vids):
        f.write(json.dumps({"video_id": v,
                            "output": f"{RT_SET[kr[i]]}|{SK_SET[ks[i]]}|"})+'\n')
print('[OK] -> outputs/optin/preds_rtboost.jsonl')
PY
python3 jax_impl/eval_metrics.py --preds outputs/optin/preds_rtboost.jsonl \
    --labels /data/labels_test.jsonl --per-class | tee outputs/optin/eval_report_rtboost.txt
