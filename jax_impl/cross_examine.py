"""三方对质: 人工 GT × 模型预测 × Gemini 盲判(训练集错误根因手术刀)。

  python3 jax_impl/cross_examine.py \
      --preds outputs/kto_prep/train_preds.jsonl \
      --gemini DATA/euno_train_v3.0.18_balanced_100k_labels.jsonl \
      --labels /data/labels_dedup.jsonl --out-dir outputs/cross_exam

原理: 模型与 Gemini 盲判是两个独立证人(不同架构/训练/信息源)。
对每条训练样本按 SubKS 分四桶:
  A 双证人赞成 GT(模型对+Gemini对)       → 标签可信,模型已会
  B 模型错,Gemini 赞成 GT                → 模型能力缺陷(训练侧靶点)
  C 模型对,Gemini 反对 GT                → Gemini 看错(难样本,正常)
  D 双证人一致反对 GT 且互相一致           → **错标重嫌**(两个独立
    判者给出同一个不同于 GT 的答案,巧合概率极低)
  D' 双证人都反对 GT 但彼此不一致          → 真难样本/边界模糊

产出:
  per_class.txt          各 SubKS 类的四桶占比表(数据质量体检报告)
  mislabel_suspects.jsonl D 桶明细(video_id/GT/模型/Gemini 三方答案)
  clean_candidates.txt    D 桶中 GT≠m 的 video_id(可清洗清单;
                          GT=m 的单列不动 —— 冻结基准下 m 懒标与
                          测试集同源,像基准比对基准重要)
纯 stdlib,宿主机可跑。
"""
import argparse
import collections
import json
import os


def load_labels(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        lab = d.get("labels", d)
        out[d["video_id"]] = (lab["role_type"], lab["sub_keyscene"])
    return out


def load_preds(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        seg = (d.get("output") or "").split("|")
        if len(seg) >= 2:
            out[d["video_id"]] = (seg[0].strip(), seg[1].strip())
    return out


def load_gemini(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        g = (d.get("gemini_output") or d).get("predictions") or {}
        rt, sk = g.get("role_type"), g.get("sub_keyscene")
        if rt and sk and d["video_id"] not in out:
            out[d["video_id"]] = (rt, sk)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="模型训练集预测 jsonl")
    ap.add_argument("--gemini", required=True, help="盲判标注 jsonl(rate≈0.37 那份)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    gt = load_labels(a.labels)
    md = load_preds(a.preds)
    gm = load_gemini(a.gemini)
    vids = [v for v in gt if v in md and v in gm]
    print(f"三方齐全样本: {len(vids)}(gt={len(gt)} model={len(md)} "
          f"gemini={len(gm)})")

    buckets = collections.defaultdict(lambda: collections.Counter())
    suspects = []
    for v in vids:
        g_sk = gt[v][1]
        m_ok = md[v][1] == g_sk
        z_ok = gm[v][1] == g_sk
        if m_ok and z_ok:
            b = "A_双证人赞成"
        elif not m_ok and z_ok:
            b = "B_模型缺陷"
        elif m_ok and not z_ok:
            b = "C_Gemini看错"
        elif md[v][1] == gm[v][1]:
            b = "D_错标重嫌"          # 双证人一致反对且答案相同
            suspects.append({"video_id": v,
                             "gt": {"rt": gt[v][0], "sk": gt[v][1]},
                             "model": {"rt": md[v][0], "sk": md[v][1]},
                             "gemini": {"rt": gm[v][0], "sk": gm[v][1]}})
        else:
            b = "E_真难样本"          # 双反对但彼此不一致
        buckets[g_sk][b] += 1
        buckets["__ALL__"][b] += 1

    cols = ["A_双证人赞成", "B_模型缺陷", "C_Gemini看错",
            "D_错标重嫌", "E_真难样本"]
    lines = [f"{'类':<10}{'n':>7}" + "".join(f"{c:>14}" for c in cols)]
    order = sorted((k for k in buckets if k != "__ALL__"),
                   key=lambda k: -buckets[k]["D_错标重嫌"]
                   / max(sum(buckets[k].values()), 1))
    for k in order + ["__ALL__"]:
        n = sum(buckets[k].values())
        row = f"{k:<10}{n:>7}"
        for c in cols:
            row += f"{buckets[k][c]/max(n,1):>13.1%} "
        lines.append(row)
    report = "\n".join(lines)
    print(report)
    open(os.path.join(a.out_dir, "per_class.txt"), "w",
         encoding="utf-8").write(report + "\n")

    with open(os.path.join(a.out_dir, "mislabel_suspects.jsonl"), "w",
              encoding="utf-8") as f:
        for s in suspects:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    n_m = sum(1 for s in suspects if s["gt"]["sk"] == "m")
    with open(os.path.join(a.out_dir, "clean_candidates.txt"), "w") as f:
        for s in suspects:
            if s["gt"]["sk"] != "m":
                f.write(s["video_id"] + "\n")
    print(f"\nD 桶(错标重嫌)共 {len(suspects)} 条,其中 GT=m {n_m} 条"
          f"(m 不入清洗清单: 与冻结测试集同源口径,像基准比对基准重要);"
          f"可清洗候选 {len(suspects)-n_m} 条 → clean_candidates.txt")
    print("判读: D 桶占比 >5% 的类 = 数据质量重灾区;B 桶占比高的类 = "
          "模型真缺陷(训练侧靶点);清洗只动 D 桶非 m 部分,重训全表验收")


if __name__ == "__main__":
    main()
