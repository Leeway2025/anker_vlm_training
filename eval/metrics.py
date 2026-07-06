"""分类指标计算(torch-free,对齐客户评测口径: P/R/ACC/混淆矩阵)。

用法:
    python -m eval.metrics --pred preds.jsonl --gt labels.jsonl
预测文件每行: {"video_id": ..., "output": "B | i | ..."} 或已解析的 {rt, sk}
GT 文件每行:  {"video_id": ..., "labels": {"role_type": "B", "sub_keyscene": "i", ...}}
"""
import json
import argparse
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.format_validator import parse_output, RT_SET, SK_SET  # noqa: E402

# 错误热区配对(training_plan.md 附录 A),单独报告
HOTSPOT_PAIRS_SK = [("h", "n"), ("k", "l"), ("s", "m")]
HOTSPOT_PAIRS_RT = [("C", "D"), ("A", "D")]
SAFETY_SK = list("qrunj")


def _prf(confusion: dict, classes) -> dict:
    """从混淆矩阵 dict[(gt,pred)]=count 算每类 P/R 与总 ACC。"""
    out, total, correct = {}, 0, 0
    for c in classes:
        tp = confusion.get((c, c), 0)
        fn = sum(v for (g, p), v in confusion.items() if g == c and p != c)
        fp = sum(v for (g, p), v in confusion.items() if g != c and p == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                  "support": tp + fn}
    total = sum(confusion.values())
    correct = sum(v for (g, p), v in confusion.items() if g == p)
    return {"per_class": out,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "total": total}


def evaluate(preds: dict, gts: dict) -> dict:
    """preds: video_id -> raw output str;gts: video_id -> (rt, sk)。"""
    conf_rt, conf_sk = defaultdict(int), defaultdict(int)
    fmt_fail = illegal = case_fixed = think_leak = 0
    n = 0
    for vid, (g_rt, g_sk) in gts.items():
        if vid not in preds:
            continue
        n += 1
        res = parse_output(preds[vid])
        if res.think_leak:
            think_leak += 1
        if res.case_fixed:
            case_fixed += 1
        if not res.ok:
            fmt_fail += 1
            # 格式失败按全错计入混淆(用特殊符号 '?')
            conf_rt[(g_rt, "?")] += 1
            conf_sk[(g_sk, "?")] += 1
            continue
        if not res.legal_combo:
            illegal += 1
        conf_rt[(g_rt, res.rt)] += 1
        conf_sk[(g_sk, res.sk)] += 1

    rt_stats = _prf(conf_rt, sorted(RT_SET))
    sk_stats = _prf(conf_sk, sorted(SK_SET))

    # KeyScene 父类聚合指标(6 大类)
    KS_GROUP = {**{c: "Normal" for c in "abcdefghijklm"},
                "n": "PropDmg", "o": "PropDmg",
                "p": "LifeThreat", "q": "LifeThreat", "r": "LifeThreat",
                "s": "Loiter", "t": "VehAnom", "u": "UnauthEntry"}
    conf_ks = defaultdict(int)
    for (g, p), v in conf_sk.items():
        conf_ks[(KS_GROUP.get(g, "?"), KS_GROUP.get(p, "?"))] += v
    ks_stats = _prf(conf_ks, sorted(set(KS_GROUP.values())))

    hotspots = {}
    for a, b in HOTSPOT_PAIRS_SK:
        hotspots[f"sk:{a}<->{b}"] = {
            f"{a}->{b}": conf_sk.get((a, b), 0),
            f"{b}->{a}": conf_sk.get((b, a), 0)}
    for a, b in HOTSPOT_PAIRS_RT:
        hotspots[f"rt:{a}<->{b}"] = {
            f"{a}->{b}": conf_rt.get((a, b), 0),
            f"{b}->{a}": conf_rt.get((b, a), 0)}

    return {
        "n_evaluated": n,
        "KeyScene_acc": ks_stats["accuracy"],
        "SubKeyScene_acc": sk_stats["accuracy"],
        "RoleType_acc": rt_stats["accuracy"],
        "format_fail_rate": round(fmt_fail / n, 4) if n else 0,
        "illegal_combo_rate": round(illegal / n, 4) if n else 0,
        "case_fixed_rate": round(case_fixed / n, 4) if n else 0,
        "think_leak_rate": round(think_leak / n, 4) if n else 0,
        "safety_recall": {c: sk_stats["per_class"].get(c, {}).get("recall")
                          for c in SAFETY_SK},
        "hotspot_confusions": hotspots,
        "rt_detail": rt_stats, "sk_detail": sk_stats,
        "confusion_sk": {f"{g}->{p}": v for (g, p), v in
                         sorted(conf_sk.items(), key=lambda kv: -kv[1])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    preds = {}
    for line in open(a.pred, encoding="utf-8"):
        d = json.loads(line)
        preds[d["video_id"]] = d.get("output") or \
            f'{d.get("rt")} | {d.get("sk")} | {d.get("desc", "")}'
    gts = {}
    for line in open(a.gt, encoding="utf-8"):
        d = json.loads(line)
        lab = d.get("labels", d)
        gts[d["video_id"]] = (lab["role_type"], lab["sub_keyscene"])

    rep = evaluate(preds, gts)
    js = json.dumps(rep, ensure_ascii=False, indent=2)
    print(js)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(js)


if __name__ == "__main__":
    main()
