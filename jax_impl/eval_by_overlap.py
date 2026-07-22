"""按"机位是否在训练集出现过"分层重打分(泄漏水分定量)。

  python3 jax_impl/eval_by_overlap.py --preds outputs/jax_5b_v2/eval_preds.jsonl \
      --test DATA/labels_test.jsonl --train DATA/labels.jsonl

零推理: 直接对已有 preds 重新分层打分。输出 重叠机位子集 vs 全新机位
子集 的 RT/SubKS/KS 三项指标 —— 两者的差 = "熟机位加成"(泄漏水分);
全新机位子集 ≈ 新装机用户的真实体验。纯 stdlib。
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.taxonomy import KS_GROUP  # noqa: E402


def cam(r):
    return (r.get("meta") or {}).get("camera_id")


def score(rows, preds):
    n = rt = sk = ks = 0
    per_sk = collections.defaultdict(lambda: [0, 0])
    for j in rows:
        seg = preds.get(j["video_id"], "").split("|")
        prt = seg[0].strip() if len(seg) >= 2 else None
        psk = seg[1].strip() if len(seg) >= 2 else None
        grt, gsk = j["labels"]["role_type"], j["labels"]["sub_keyscene"]
        n += 1
        rt += (prt == grt)
        sk += (psk == gsk)
        ks += (psk is not None and gsk in KS_GROUP and psk in KS_GROUP
               and KS_GROUP[psk] == KS_GROUP[gsk])
        per_sk[gsk][0] += 1
        per_sk[gsk][1] += (psk == gsk)
    return n, rt, sk, ks, per_sk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", required=True)
    a = ap.parse_args()
    preds = {j["video_id"]: j["output"] for j in
             map(json.loads, open(a.preds, encoding="utf-8"))}
    train_cams = {cam(json.loads(l)) for l in open(a.train, encoding="utf-8")}
    te = [json.loads(l) for l in open(a.test, encoding="utf-8")]
    seen = [j for j in te if cam(j) in train_cams]
    fresh = [j for j in te if cam(j) not in train_cams]
    print(f"test 总 {len(te)} = 重叠机位 {len(seen)} + 全新机位 {len(fresh)}\n")
    print(f"{'子集':<10}{'n':>6}{'RT':>9}{'SubKS':>9}{'KS父类':>9}")
    res = {}
    for name, rows in (("重叠机位", seen), ("全新机位", fresh), ("全量", te)):
        n, rt, sk, ks, per = score(rows, preds)
        res[name] = (sk / max(n, 1), per)
        print(f"{name:<10}{n:>6}{rt/max(n,1):>9.2%}{sk/max(n,1):>9.2%}"
              f"{ks/max(n,1):>9.2%}")
    gap = res["重叠机位"][0] - res["全新机位"][0]
    print(f"\n熟机位加成(SubKS): {gap:+.2%}"
          + ("  ⚠️ 显著: 账面指标含泄漏水分,交付口径建议以全新机位子集为准"
             if abs(gap) > 0.03 else "  (差异不大,泄漏影响有限)"))
    print("\n== 全新机位子集 各类 recall(真实泛化画像,n≥30 的类)==")
    per = res["全新机位"][1]
    for skl in sorted(per, key=lambda k: per[k][1] / max(per[k][0], 1)):
        c, ok = per[skl]
        if c >= 30:
            print(f"  [{skl}] n={c:<5} recall={ok/max(c,1):.1%}")


if __name__ == "__main__":
    main()
