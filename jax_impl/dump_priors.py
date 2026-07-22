"""从 labels.jsonl 导出 RT/SubKS 先验分布(先验校正的输入件)。

  python3 jax_impl/dump_priors.py --labels 原始全量标注.jsonl --out priors_natural.json
  python3 jax_impl/dump_priors.py --labels DATA/labels.jsonl --out priors_train.json
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rt, sk = collections.Counter(), collections.Counter()
    for l in open(a.labels, encoding="utf-8"):
        j = json.loads(l)
        rt[j["labels"]["role_type"]] += 1
        sk[j["labels"]["sub_keyscene"]] += 1
    out = {"rt": {k: v / sum(rt.values()) for k, v in rt.items()},
           "sk": {k: v / sum(sk.values()) for k, v in sk.items()}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[OK] {a.out}  rt={dict(rt)}  sk 类数={len(sk)}")


if __name__ == "__main__":
    main()
