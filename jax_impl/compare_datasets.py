"""train/test 数据集结构对比(分布差异定量 + 泄漏/口径线索)。

  python3 jax_impl/compare_datasets.py --train DATA/labels.jsonl \
      --test DATA/labels_test.jsonl

输出: ①SubKS/RT 类分布对照与占比差(过拟合到 train 口味时,占比差大的
类就是重灾区)②camera 重叠(>0 即 train/test 泄漏或同源线索)③meta
字段覆盖率对照(annotation_version 等口径字段不一致 = 两批标注)
④description 统计(均长/词汇重叠,口径漂移的粗指纹)。纯 stdlib。
"""
import argparse
import collections
import json


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def dist(rows, key):
    c = collections.Counter(r["labels"][key] for r in rows)
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    tr, te = load(a.train), load(a.test)
    print(f"train={len(tr)}  test={len(te)}\n")

    for key in ("sub_keyscene", "role_type"):
        dtr, ctr = dist(tr, key)
        dte, cte = dist(te, key)
        ks = sorted(set(dtr) | set(dte),
                    key=lambda k: -abs(dtr.get(k, 0) - dte.get(k, 0)))
        print(f"== {key} 分布对照(按占比差排序)==")
        print(f"{'类':<4}{'train':>9}{'test':>9}{'占比差':>9}")
        for k in ks:
            d = dtr.get(k, 0) - dte.get(k, 0)
            flag = "  ⚠️" if abs(d) > 0.05 else ""
            print(f"{k:<4}{dtr.get(k,0):>8.1%}{dte.get(k,0):>8.1%}"
                  f"{d:>+8.1%}{flag}")
        only_tr = set(ctr) - set(cte)
        only_te = set(cte) - set(ctr)
        if only_tr:
            print("  仅 train 有:", sorted(only_tr))
        if only_te:
            print("  仅 test 有:", sorted(only_te))
        print()

    cam = lambda r: (r.get("meta") or {}).get("camera_id")
    cams_tr = {cam(r) for r in tr if cam(r)}
    cams_te = {cam(r) for r in te if cam(r)}
    inter = cams_tr & cams_te
    print(f"== camera == train {len(cams_tr)} 个 / test {len(cams_te)} 个 / "
          f"重叠 {len(inter)} 个"
          + ("  ⚠️ 存在重叠: 同机位横跨训练与测试(泄漏或抽样同源)"
             if inter else "  ✅ 无重叠"))
    if inter:
        n_te_overlap = sum(1 for r in te if cam(r) in inter)
        print(f"   test 中来自重叠 camera 的样本: {n_te_overlap} 条"
              f"({n_te_overlap/len(te):.1%})")

    print("\n== meta 字段覆盖率 ==")
    keys = sorted({k for r in tr + te for k in (r.get("meta") or {})})
    for k in keys:
        f_tr = sum(1 for r in tr if (r.get("meta") or {}).get(k) is not None) / len(tr)
        f_te = sum(1 for r in te if (r.get("meta") or {}).get(k) is not None) / len(te)
        vals_tr = collections.Counter(
            str((r.get("meta") or {}).get(k)) for r in tr[:5000]
            if k == "annotation_version")
        note = f"  train 版本分布 {dict(vals_tr)}" if vals_tr else ""
        print(f"  {k:<22} train {f_tr:>6.1%}  test {f_te:>6.1%}{note}")

    print("\n== description 指纹 ==")
    for name, rows in (("train", tr), ("test", te)):
        ls = [len(str(r["labels"].get("description", "")).split())
              for r in rows]
        print(f"  {name}: 平均 {sum(ls)/max(len(ls),1):.1f} 词, "
              f"中位 {sorted(ls)[len(ls)//2]} 词")
    vt = collections.Counter(w for r in tr[:3000]
                             for w in str(r["labels"]["description"]).lower().split())
    ve = collections.Counter(w for r in te[:3000]
                             for w in str(r["labels"]["description"]).lower().split())
    top_tr = {w for w, _ in vt.most_common(200)}
    top_te = {w for w, _ in ve.most_common(200)}
    j = len(top_tr & top_te) / max(len(top_tr | top_te), 1)
    print(f"  高频词 Jaccard 重叠: {j:.0%}"
          + ("(<60%: 描述文风/口径差异明显)" if j < 0.6 else " ✅"))


if __name__ == "__main__":
    main()
