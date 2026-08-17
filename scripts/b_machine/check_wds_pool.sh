#!/bin/bash
# 1M P4a: WDS 覆盖检查(labels 的视频都能在分片索引里找到吗)+ 读取吞吐抽测
# 用法: bash scripts/check_wds_pool.sh /path/labels_1m_v1.jsonl <WDS目录>
set -e
cd "$(dirname "$0")/.."
LB="${1:?}"; WD="${2:?用法: check_wds_pool.sh <labels> <wds目录>}"
LB="$LB" WD="$WD" python3 - <<'PY'
import json, os, random, sys, time
sys.path.insert(0, '.')
from data.euno_wds import read_json
lb, wd = os.environ['LB'], os.environ['WD'].rstrip('/')
idx = read_json(f'{wd}/index.json')
members = set(idx) if isinstance(idx, (list, set)) else set(idx.keys())
vids = [json.loads(l)['video_id'] for l in open(lb, encoding='utf-8')]
miss = [v for v in vids if v not in members and f'{v}.pyd' not in members]
print(f'覆盖: {len(vids)-len(miss)}/{len(vids)}(缺 {len(miss)});缺样例: {miss[:3]}')
random.seed(0)
import tarfile
shards = sorted({s for s in os.listdir(wd) if s.endswith('.tar')})
t0, nb = time.time(), 0
for s in random.sample(shards, min(3, len(shards))):
    with tarfile.open(f'{wd}/{s}') as tf:
        for m in tf.getmembers()[:64]:
            nb += len(tf.extractfile(m).read())
dt = time.time() - t0
print(f'吞吐抽测: {nb/1e6:.0f}MB / {dt:.1f}s = {nb/1e6/dt:.0f} MB/s '
      f'(训练需求参考: 100k配方约需 ≥80MB/s 不喂饿 TPU)')
PY
