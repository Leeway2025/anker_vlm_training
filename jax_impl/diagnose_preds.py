"""评测结果诊断: 预测/GT 字母分布 + 混淆映射 + 训练/测试分布对照。

  python jax_impl/diagnose_preds.py --preds outputs/eval_preds.jsonl \
      --labels DATA/labels_test.jsonl [--train-labels DATA/labels.jsonl] \
      [--meta outputs/jax_5b/train_meta.json]

判读指南(2026-07-21 现场经验):
  - TOP 混淆呈稳定一对一映射(gt=m 几乎全预测成同组某字母)
    → 训练/测试标签的字母口径不一致(taxonomy 版本/标注来源),非模型问题;
  - 预测直方图高度集中在三五个字母、与视频内容无关
    → 模型退化为先验输出(查视觉链路/训练标签噪声);
  - train 与 test 的 GT 直方图形状差异巨大 → 数据不同源,先问数据方。
纯 stdlib,宿主机 python3 直接跑。
"""
import argparse
import collections
import json


def hist_pct(counter, total):
    return {k: f"{v}({100.0*v/max(total,1):.1f}%)"
            for k, v in sorted(counter.items(), key=lambda x: -x[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True, help="测试集 GT")
    ap.add_argument("--train-labels", help="训练集 GT(可选,分布对照)")
    ap.add_argument("--meta", help="train_meta.json(可选,best 选点)")
    ap.add_argument("--topk", type=int, default=15)
    a = ap.parse_args()

    pred = {j["video_id"]: j["output"] for j in
            map(json.loads, open(a.preds, encoding="utf-8"))}
    gt = [json.loads(l) for l in open(a.labels, encoding="utf-8")]

    def parse(p):
        seg = p.split("|")
        return (seg[0].strip(), seg[1].strip()) if len(seg) >= 3 else (None, None)

    ph_rt, ph_sk = collections.Counter(), collections.Counter()
    cf_rt, cf_sk = collections.Counter(), collections.Counter()
    n_hit = 0
    for j in gt:
        p = pred.get(j["video_id"])
        if p is None:
            continue
        n_hit += 1
        prt, psk = parse(p)
        if psk is None:
            continue
        ph_rt[prt] += 1
        ph_sk[psk] += 1
        cf_rt[(j["labels"]["role_type"], prt)] += 1
        cf_sk[(j["labels"]["sub_keyscene"], psk)] += 1
    gh_rt = collections.Counter(j["labels"]["role_type"] for j in gt)
    gh_sk = collections.Counter(j["labels"]["sub_keyscene"] for j in gt)

    print(f"== 覆盖: preds {len(pred)} 条, gt {len(gt)} 条, 匹配 {n_hit} ==")
    print("\n== SubKS 字母分布 ==")
    print("PRED:", hist_pct(ph_sk, n_hit))
    print("GT:  ", hist_pct(gh_sk, len(gt)))
    print("模型从不输出的 GT 字母:", sorted(set(gh_sk) - set(ph_sk)))
    print("\n== RT 字母分布 ==")
    print("PRED:", hist_pct(ph_rt, n_hit))
    print("GT:  ", hist_pct(gh_rt, len(gt)))

    print(f"\n== SubKS TOP{a.topk} 混淆(gt→pred, 含对角) ==")
    for (g, p), c in cf_sk.most_common(a.topk):
        mark = "✓" if g == p else ("→ 一对一?" if c > 0.5 * gh_sk[g] else "")
        print(f"  {g} → {p}  {c:5d}  ({100.0*c/max(gh_sk[g],1):.0f}% of gt-{g}) {mark}")
    print(f"\n== RT TOP{min(a.topk,10)} 混淆 ==")
    for (g, p), c in cf_rt.most_common(min(a.topk, 10)):
        print(f"  {g} → {p}  {c:5d}  ({100.0*c/max(gh_rt[g],1):.0f}% of gt-{g})"
              f" {'✓' if g == p else ''}")

    if a.train_labels:
        th = collections.Counter(
            json.loads(l)["labels"]["sub_keyscene"]
            for l in open(a.train_labels, encoding="utf-8"))
        tot = sum(th.values())
        print("\n== 训练集 GT SubKS 分布(与测试集形状对照) ==")
        print("TRAIN:", hist_pct(th, tot))
        only_test = sorted(set(gh_sk) - set(th))
        if only_test:
            print("⚠️ 测试集有、训练集没有的字母:", only_test)
    if a.meta:
        m = json.load(open(a.meta, encoding="utf-8"))
        hist = m.get("loss_history", [])
        print(f"\n== 训练元信息 == best_val={m.get('best_val')} "
              f"commit={m.get('code_commit')} "
              f"loss 首/尾: {hist[0]:.3f}/{hist[-1]:.3f}" if hist else m)


if __name__ == "__main__":
    main()
