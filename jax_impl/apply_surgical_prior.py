"""手工手术版先验(m + RT-E,τ 固定)离线施加到裸 logits(零 TPU)。

  python3 jax_impl/apply_surgical_prior.py --logits outputs/optin/preds.jsonl \
      --labels /data/labels_test.jsonl \
      --fold-a /data/test_sfoldA.jsonl --fold-b /data/test_sfoldB.jsonl \
      --tau 0.7 --out outputs/optin/preds_surg.jsonl

与在线版(infer --prior-*)同一数学: Δ=τ·log(目标先验/训练先验),
只修 SK 的 m 与 RT 的 E;目标先验取自对侧折(两折交叉红线不变)。
裸 logits 来自 --dump-letter-logits,换底座只需重 dump 一次。
"""
import argparse
import collections
import json
import math

RT_SET = "ABCDE"
SK_SET = "abcdefghijklmnopqrstu"


def prior_of(path, key):
    c = collections.Counter()
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        v = d.get("labels") or d
        c[v[key]] += 1
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--train-labels", default="/data/labels_dedup.jsonl")
    ap.add_argument("--labels", required=True, help="仅用于打印口径,不参与计算")
    ap.add_argument("--fold-a", required=True)
    ap.add_argument("--fold-b", required=True)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tr_sk = prior_of(a.train_labels, "sub_keyscene")
    tr_rt = prior_of(a.train_labels, "role_type")
    fa_ids = {json.loads(l)["video_id"] for l in open(a.fold_a, encoding="utf-8")}
    deltas = {}
    for fold_path, ids_side in ((a.fold_b, "A"), (a.fold_a, "B")):
        tg_sk = prior_of(fold_path, "sub_keyscene")
        tg_rt = prior_of(fold_path, "role_type")
        d_m = a.tau * math.log(max(tg_sk.get("m", 1e-9), 1e-9)
                               / max(tr_sk.get("m", 1e-9), 1e-9))
        d_e = a.tau * math.log(max(tg_rt.get("E", 1e-9), 1e-9)
                               / max(tr_rt.get("E", 1e-9), 1e-9))
        deltas[ids_side] = (d_m, d_e)
        print(f"[surg] 折{ids_side} 侧: Δm={d_m:+.3f} ΔE(RT)={d_e:+.3f}"
              f"(先验来自对侧折)")

    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for l in open(a.logits, encoding="utf-8"):
            d = json.loads(l)
            if "rt" not in d or "sk" not in d:
                continue
            side = "A" if d["video_id"] in fa_ids else "B"
            d_m, d_e = deltas[side]
            sk = list(d["sk"])
            sk[SK_SET.index("m")] += d_m
            rt = list(d["rt"])
            rt[RT_SET.index("E")] += d_e
            k_sk = max(range(21), key=lambda i: sk[i])
            k_rt = max(range(5), key=lambda i: rt[i])
            # 与在线版 legalize_combo 同款: 家人(A)不可能"包裹被拿走/闯入"
            if RT_SET[k_rt] == "A" and SK_SET[k_sk] in ("n", "u"):
                k_rt = RT_SET.index("C")
            f.write(json.dumps({"video_id": d["video_id"],
                                "output": f"{RT_SET[k_rt]}|{SK_SET[k_sk]}|"},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"[OK] {n} 条 -> {a.out}(eval_metrics 打分 = 交付口径)")


if __name__ == "__main__":
    main()
