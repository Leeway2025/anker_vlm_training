#!/usr/bin/env python3
"""从 test 切一块当 val(快照式,一次定死).

规则(v2, 08-07 用户拍板不去噪——"噪声"多为客户口音,阅卷按原样标签):
  1. 全量 11022 参与, 不剔错标清单(val 使命=预测客户打分, 与阅卷口径一致)
  2. 整组切: 同源视频(去掉 _segment_N)的所有片段同侧,防近重复横跨 val/report
  3. 分层: 多次随机试验,选 RT×SubKS 联合边际最接近全量 test 的一份
  4. 产物: test_val_ids_v2.txt(约 TARGET 条)+ labels_train_plus_testval_v2.jsonl
     (=700k 池 + 仅 val 那部分 test 行; report 部分不进文件, 物理无泄漏)
"""
import json, random, re, collections

TEST = "/data/labels_test.jsonl"
POOL = "/data/pool_700k_final_labels.jsonl"
EXCL = []
OUT_IDS = "/data/test_val_ids_v2.txt"
OUT_COMBINED = "/data/labels_train_plus_testval_v2.jsonl"
TARGET = 2000
TRIALS = 500

recs = [json.loads(l) for l in open(TEST, encoding="utf-8")]
excl = set()
for f in EXCL:
    excl |= set(open(f).read().split())
clean = [r for r in recs if r["video_id"] not in excl]
print(f"test 全量 {len(recs)}  去噪后 {len(clean)}  (剔 {len(recs)-len(clean)})")

# 红线补丁: --val-ids 用 .split() 解析, id 含 unicode 空白(如 U+202F)会被切碎
# → 匹配失败 → 该 test 行漏进 train。这类 id 一律不进 val 候选(也就不进 combined)
n0 = len(clean)
clean = [r for r in clean if len(r["video_id"].split()) == 1]
print(f"剔除含空白id {n0-len(clean)} 条 → 候选 {len(clean)}")

def cls(r):
    return (r["labels"].get("role_type", "?"), r["labels"].get("sub_keyscene", "?"))

def base(vid):
    return re.sub(r"_segment_\d+$", "", vid)

groups = collections.defaultdict(list)
for r in clean:
    groups[base(r["video_id"])].append(r)
gkeys = sorted(groups)
print(f"源视频组 {len(gkeys)} 个, 平均 {len(clean)/len(gkeys):.2f} 段/组")

total = collections.Counter(cls(r) for r in clean)
n_clean = len(clean)

best = None
for t in range(TRIALS):
    rng = random.Random(1000 + t)
    ks = gkeys[:]
    rng.shuffle(ks)
    val, n = [], 0
    for k in ks:
        if n >= TARGET:
            break
        val += groups[k]
        n += len(groups[k])
    cnt = collections.Counter(cls(r) for r in val)
    # L1 距离: val 边际 vs 全量去噪 test 边际
    d = sum(abs(cnt.get(c, 0) / len(val) - total[c] / n_clean) for c in total)
    if best is None or d < best[0]:
        best = (d, t, val)

d, t, val = best
val_ids = sorted(r["video_id"] for r in val)
print(f"选中 trial#{t}  val={len(val)} 条  L1边际距离={d:.4f}")

vc = collections.Counter(r["labels"]["role_type"] for r in val)
tc = collections.Counter(r["labels"]["role_type"] for r in clean)
print("RT 边际  val% / test%:")
for k in sorted(tc):
    print(f"  {k}: {100*vc.get(k,0)/len(val):5.1f} / {100*tc[k]/n_clean:5.1f}")
sq = set("qrunj")
print(f"安全关键 SubKS(qrunj): val 内 {sum(1 for r in val if r['labels'].get('sub_keyscene') in sq)} 条")

with open(OUT_IDS, "w") as f:
    f.write("\n".join(val_ids) + "\n")

vset = set(val_ids)
n_pool = 0
with open(OUT_COMBINED, "w", encoding="utf-8") as out:
    for l in open(POOL, encoding="utf-8"):
        out.write(l)
        n_pool += 1
    for r in recs:                      # 只写 val 命中的 test 行
        if r["video_id"] in vset:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"combined = 700k池 {n_pool} + test-val {len(vset)} → {OUT_COMBINED}")
print(f"ids → {OUT_IDS}")
