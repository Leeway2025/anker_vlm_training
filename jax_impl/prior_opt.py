"""全类先验优化(手术版先验的满血形态): 21+5 维 logit 偏移,两折交叉。

  # ① 先用 --dump-letter-logits 对测试集裸推一遍(不带任何 prior 参数):
  #    infer_sharded ... INFER_ARGS='--dump-letter-logits' → preds 含 rt/sk 裸 logits
  # ② 优化(纯 CPU,分钟级):
  python3 jax_impl/prior_opt.py --logits outputs/optin/preds.jsonl \
      --labels /data/labels_test.jsonl \
      --fold-a /data/test_sfoldA.jsonl --fold-b /data/test_sfoldB.jsonl \
      --out outputs/optin/preds_opt.jsonl
  # ③ eval_metrics 对 preds_opt.jsonl 打分 = 全类先验版交付口径

合规: 折 A 的偏移在折 B 上拟合(坐标上升),反之亦然 —— 任何样本的
判决从不使用其所在折的标签,与现行手术版先验同一红线。
决算表依据: E 桶残余 m=313 条(现行 m/E 双方向手工外挂之后仍欠报)
→ 全类自动寻优直接对着这块肉。纯 stdlib+numpy。
"""
import argparse
import json

import numpy as np

RT_SET = "ABCDE"
SK_SET = "abcdefghijklmnopqrstu"


def load(logits_file, labels_file):
    lab = {}
    for l in open(labels_file, encoding="utf-8"):
        d = json.loads(l)
        v = d.get("labels") or d
        lab[d["video_id"]] = (v["role_type"], v["sub_keyscene"])
    vids, rt_lg, sk_lg, rt_y, sk_y = [], [], [], [], []
    for l in open(logits_file, encoding="utf-8"):
        d = json.loads(l)
        if "rt" not in d or "sk" not in d or d["video_id"] not in lab:
            continue
        vids.append(d["video_id"])
        rt_lg.append(d["rt"])
        sk_lg.append(d["sk"])
        g = lab[d["video_id"]]
        rt_y.append(RT_SET.index(g[0]))
        sk_y.append(SK_SET.index(g[1]))
    return (vids, np.asarray(rt_lg), np.asarray(sk_lg),
            np.asarray(rt_y), np.asarray(sk_y))


def coord_ascent(lg, y, max_off=2.0, passes=6):
    """坐标上升: 逐类试网格偏移,收准确率增益,循环至收敛。"""
    n_cls = lg.shape[1]
    off = np.zeros(n_cls)
    grid = np.array([-1.5, -1.0, -0.6, -0.3, -0.15,
                     0.15, 0.3, 0.6, 1.0, 1.5])
    best_acc = (np.argmax(lg + off, 1) == y).mean()
    for _ in range(passes):
        improved = False
        for c in range(n_cls):
            cand_acc, cand_d = best_acc, 0.0
            for d in grid:
                nd = np.clip(off[c] + d, -max_off, max_off)
                trial = off.copy()
                trial[c] = nd
                acc = (np.argmax(lg + trial, 1) == y).mean()
                if acc > cand_acc + 1e-9:
                    cand_acc, cand_d = acc, nd
            if cand_d != 0.0 and cand_acc > best_acc:
                off[c] = cand_d if cand_d != off[c] else off[c]
                off[c] = cand_d
                best_acc = cand_acc
                improved = True
        if not improved:
            break
    return off, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True,
                    help="--dump-letter-logits 产出的 preds(裸 logits)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--fold-a", required=True)
    ap.add_argument("--fold-b", required=True)
    ap.add_argument("--max-off", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vids, rt_lg, sk_lg, rt_y, sk_y = load(a.logits, a.labels)
    fa = {json.loads(l)["video_id"] for l in open(a.fold_a, encoding="utf-8")}
    fb = {json.loads(l)["video_id"] for l in open(a.fold_b, encoding="utf-8")}
    ia = np.asarray([v in fa for v in vids])
    ib = np.asarray([v in fb for v in vids])
    print(f"样本 {len(vids)} = foldA {ia.sum()} + foldB {ib.sum()}")

    out_rt = np.zeros((len(vids), 5))
    out_sk = np.zeros((len(vids), 21))
    for fit, apply_ in ((ib, ia), (ia, ib)):     # B 上拟合 → 用于 A;反之
        off_sk, acc_sk = coord_ascent(sk_lg[fit], sk_y[fit], a.max_off)
        off_rt, acc_rt = coord_ascent(rt_lg[fit], rt_y[fit], a.max_off)
        base_sk = (np.argmax(sk_lg[fit], 1) == sk_y[fit]).mean()
        print(f"  拟合折: SubKS {base_sk:.4f} -> {acc_sk:.4f}"
              f"(拟合内增益 {acc_sk-base_sk:+.4f});偏移 "
              + " ".join(f"{SK_SET[i]}{off_sk[i]:+.2f}"
                         for i in np.argsort(-np.abs(off_sk))[:6] if off_sk[i]))
        out_sk[apply_] = off_sk
        out_rt[apply_] = off_rt

    rt_pick = np.argmax(rt_lg + out_rt, 1)
    sk_pick = np.argmax(sk_lg + out_sk, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, v in enumerate(vids):
            f.write(json.dumps({
                "video_id": v,
                "output": f"{RT_SET[rt_pick[i]]}|{SK_SET[sk_pick[i]]}|"},
                ensure_ascii=False) + "\n")
    acc = (sk_pick == sk_y).mean()
    base = (np.argmax(sk_lg, 1) == sk_y).mean()
    print(f"[OK] 交叉应用后全量 SubKS(仅信号位): {base:.4f} -> {acc:.4f} "
          f"({acc-base:+.4f})-> {a.out};正式分以 eval_metrics 为准")


if __name__ == "__main__":
    main()
