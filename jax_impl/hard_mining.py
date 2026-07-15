"""难例挖掘(JAX 路线独立版,零 torch 依赖;逻辑对齐 training/hard_mining.py)。

  python jax_impl/hard_mining.py --preds preds.jsonl --labels labels.jsonl \
      --out sw.json [--weight 3.0] [--max-class-inflation 1.5]

错例权重 w,按 SubKS 类做质量膨胀上限:
  类 c 有 n_c 条、错例 k_c 条 → w_c = min(w, 1 + (cap-1)·n_c/k_c)
"""
import argparse
import json
import re
from collections import defaultdict


def parse_output(text):
    m = re.match(r"\s*([A-E])\s*\|\s*([a-u])\s*\|", text or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight", type=float, default=3.0)
    ap.add_argument("--max-class-inflation", type=float, default=1.5)
    a = ap.parse_args()

    preds = {j["video_id"]: j["output"] for j in
             (json.loads(l) for l in open(a.preds, encoding="utf-8"))}
    gts = {}
    for l in open(a.labels, encoding="utf-8"):
        j = json.loads(l)
        gts[j["video_id"]] = (j["labels"]["role_type"],
                              j["labels"]["sub_keyscene"])

    n_c, k_c, wrong = defaultdict(int), defaultdict(int), {}
    missing = 0
    for vid, (rt, sk) in gts.items():
        n_c[sk] += 1
        if vid not in preds:
            missing += 1
            continue
        prt, psk = parse_output(preds[vid])
        if prt != rt or psk != sk:
            k_c[sk] += 1
            wrong[vid] = sk

    cap = a.max_class_inflation
    w_class = {c: min(a.weight, 1 + (cap - 1) * n_c[c] / k)
               for c, k in k_c.items() if k}
    weights = {vid: (round(w_class[sk], 3) if vid in wrong else 1.0)
               for vid, (_, sk) in gts.items()}
    json.dump(weights, open(a.out, "w"), indent=1)
    print(f"[hard-mining] gts={len(gts)} wrong={len(wrong)} "
          f"missing_pred={missing} class_caps={dict(list(w_class.items())[:5])}"
          f" -> {a.out}")


if __name__ == "__main__":
    main()
