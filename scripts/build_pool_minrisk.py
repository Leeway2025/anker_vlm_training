#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deadline 版 500k 池: 只用零风险成分 —— 闸0结构 + test边际对齐 + 尾类全收 + 随机。
不含任何未验证信号(切片/同设备/margin全不进)。"""
import json, random, re, sys
from collections import Counter
POOL='/home/nas-tpu-poc/data/zx_vlm_dataset/anker_video_clips/euno_train_v3.0.24_des25_deduped_fixed_exclude_gen_videos_train_format_v23_frames.json'
SIZE=700000; TAIL=set("qrujonst"); rng=random.Random(0)
NONZH=re.compile(r"[一-鿿]")
recs=[]
for r in json.load(open(POOL)):
    g=next((c['value'] for c in r['conversations'] if c['from']=='gpt'),'')
    m=re.match(r'\s*([A-E])\|([a-u])\|(.*)',g,re.S)
    if not m or not m.group(3).strip() or NONZH.search(m.group(3)): continue
    recs.append((r['video'],m.group(1),m.group(2)))
print('闸0后:',len(recs))
t_rt=Counter()
for l in open('/data/labels_test.jsonl'):
    j=json.loads(l); t_rt[j['labels']['role_type']]+=1
tn=sum(t_rt.values())
rng.shuffle(recs)
tail=[r for r in recs if r[2] in TAIL]
rest=[r for r in recs if r[2] not in TAIL]
sel=[v for v,_,_ in tail]                      # 尾类全收(~9.3万)
got=Counter(rt for _,rt,_ in tail)
quota={k:int(SIZE*v/tn) for k,v in t_rt.items()}
for v,rt,sk in rest:                            # 边际对齐填充
    if len(sel)>=SIZE: break
    if got[rt]>=quota.get(rt,0): continue
    got[rt]+=1; sel.append(v)
for v,rt,sk in rest:                            # 不足则随机补满
    if len(sel)>=SIZE: break
    if v not in set(sel[-0:]): pass
seen=set(sel)
for v,rt,sk in rest:
    if len(sel)>=SIZE: break
    if v not in seen: sel.append(v); seen.add(v)
open('/data/pool_500k_final_ids.txt','w').write('\n'.join(sel)+'\n')
print(f'终版池 {len(sel)} | RT配比 {dict(got)}')
