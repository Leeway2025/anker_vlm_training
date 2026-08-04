#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 A↔D 混淆错例清单,供肉眼判"缺帧 vs 缺判别力"。
关键判据: 并排 GT 描述 与 模型自产描述——
  · 模型描述已含正确身份线索却仍判错 → 判别力/偏见问题(重切帧无用);
  · 模型描述含糊/漏掉线索          → 可能缺帧(→ 值得 Phase B 稠密帧核实)。
用法: python3 scripts/dump_ad_errors.py [--preds outputs/optin/preds.jsonl]
      产物: outputs/diag/ad_errors.jsonl(全量 971 条) + 屏幕抽样(按自信度降序)
"""
import argparse, json, math, os

RT_SET = "ABCDE"


def softmax(xs):
    m = max(xs); es = [math.exp(x - m) for x in xs]; s = sum(es)
    return [e / s for e in es]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="outputs/optin/preds.jsonl")
    ap.add_argument("--labels", default="/data/labels_test.jsonl")
    ap.add_argument("--out", default="outputs/diag/ad_errors.jsonl")
    ap.add_argument("--sample", type=int, default=10, help="每方向屏幕抽样条数")
    args = ap.parse_args()

    gt = {}
    for l in open(args.labels, encoding="utf-8"):
        d = json.loads(l); lb = d.get("labels") or d
        gt[d["video_id"]] = (lb["role_type"], lb["sub_keyscene"], lb.get("description", ""))

    rows = []
    for l in open(args.preds, encoding="utf-8"):
        d = json.loads(l); vid = d["video_id"]
        if vid not in gt or "rt" not in d:
            continue
        g_rt = gt[vid][0]
        probs = softmax(d["rt"])
        pi = max(range(5), key=lambda i: probs[i])
        p_rt = RT_SET[pi]
        if {g_rt, p_rt} != {"A", "D"} or g_rt == p_rt:
            continue
        # 模型自产 output = "RT|SK|desc"
        parts = (d.get("output") or "").split("|", 2)
        pred_sk = parts[1] if len(parts) > 1 else "?"
        pred_desc = parts[2] if len(parts) > 2 else ""
        rows.append({
            "video_id": vid,
            "dir": f"{g_rt}→{p_rt}",           # 真→预测
            "gt_rt": g_rt, "pred_rt": p_rt,
            "p_pred": round(probs[pi], 3), "p_gt": round(probs[RT_SET.index(g_rt)], 3),
            "gt_sk": gt[vid][1], "pred_sk": pred_sk,
            "gt_desc": gt[vid][2], "pred_desc": pred_desc,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows.sort(key=lambda r: -r["p_pred"])           # 最自信答错的排前(最诊断)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    nAD = sum(1 for r in rows if r["dir"] == "A→D")
    nDA = len(rows) - nAD
    print(f"[out] {len(rows)} 条 A↔D 错例 -> {args.out}  (A→D {nAD} | D→A {nDA})")
    print("     排序: 预测类置信度降序(越靠前=模型越'自信答错')\n")

    for direction in ("A→D", "D→A"):
        sub = [r for r in rows if r["dir"] == direction][:args.sample]
        head = ("A→D  真=A(住户/家人) 被判 D(陌生/快递)" if direction == "A→D"
                else "D→A  真=D(陌生/快递) 被判 A(住户/家人)")
        print(f"================ {head} ================")
        for r in sub:
            print(f"\n[{r['video_id']}]  p(判错)={r['p_pred']} p(真类)={r['p_gt']}  "
                  f"SK 真={r['gt_sk']}/判={r['pred_sk']}")
            print(f"  GT 描述 : {r['gt_desc'][:200]}")
            print(f"  模型描述: {r['pred_desc'][:200]}")


if __name__ == "__main__":
    main()
