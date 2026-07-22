"""标注口径盲审 —— 结论计算器。

  python3 jax_impl/audit_score.py --worksheet outputs/audit/worksheet.csv \
      --key outputs/audit/answer_key.json \
      [--report outputs/jax_5b_v2/eval/report.json]  # 附带修正指标外推

判定列取值: 1 / 2(哪个候选对)/ both / neither / unsure。
输出: 各分层的 模型胜率 / GT 胜率 / 双对率;m_disputed 层若模型胜率
显著,按该层占比外推"审计修正后 SubKS"。对照组用于校准审计者
(control 层 GT 胜率应接近 100%,否则审计本身不可信)。
"""
import argparse
import collections
import csv
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worksheet", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--report", help="eval report.json(外推修正指标用)")
    ap.add_argument("--m-total", type=int, default=3036,
                    help="测试集 gt=m 总数(默认取 v2 口径)")
    ap.add_argument("--m-missed", type=int, default=1226,
                    help="gt=m 被判其它类的条数")
    ap.add_argument("--n-total", type=int, default=11022)
    a = ap.parse_args()
    key = json.load(open(a.key, encoding="utf-8"))
    stats = collections.defaultdict(collections.Counter)
    n_filled = 0
    with open(a.worksheet, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            verdict = (row.get("判定(1/2/both/neither/unsure)") or "").strip().lower()
            if not verdict:
                continue
            n_filled += 1
            k = key[str(int(row["idx"]))]
            st, gt_is = k["stratum"], k["gt_is"]
            if verdict in ("1", "2"):
                stats[st]["gt_win" if int(verdict) == gt_is else "model_win"] += 1
            elif verdict in ("both", "neither", "unsure"):
                stats[st][verdict] += 1
            stats[st]["n"] += 1
    if not n_filled:
        raise SystemExit("判定列全空 —— 请先人工填写 worksheet.csv")
    print(f"已判 {n_filled} 条\n")
    print(f"{'分层':<12}{'n':>4}{'GT胜':>6}{'模型胜':>7}{'都对':>6}{'都错':>6}{'不确定':>7}")
    for st, c in stats.items():
        print(f"{st:<12}{c['n']:>4}{c['gt_win']:>6}{c['model_win']:>7}"
              f"{c['both']:>6}{c['neither']:>6}{c['unsure']:>7}")
    ctrl = stats.get("control", {})
    if ctrl and ctrl.get("n"):
        ok = (ctrl.get("gt_win", 0) + ctrl.get("both", 0)) / ctrl["n"]
        print(f"\n[校准] 对照组认可率 {ok:.0%}"
              + ("(<80%: 审计口径与 GT 体系性分歧,结论慎用)" if ok < 0.8 else " ✅"))
    md = stats.get("m_disputed", {})
    if md and md.get("n"):
        mw = (md.get("model_win", 0) + md.get("both", 0)) / md["n"]
        recovered = int(a.m_missed * mw)
        print(f"\n[m 层] 模型胜+双对 = {mw:.0%} → 外推: gt=m 的 {a.m_missed} 条"
              f"\"错误\"中约 {recovered} 条实为可接受")
        if a.report:
            rep = json.load(open(a.report, encoding="utf-8"))
            adj = rep["SubKeyScene_acc"] + recovered / a.n_total
            print(f"[外推] 审计修正后 SubKS ≈ {adj:.4f}"
                  f"(原 {rep['SubKeyScene_acc']:.4f})")
        if mw >= 0.4:
            print("[结论] m 类存在系统性懒标 → 建议: ①测试集 GT 修订该层 "
                  "②训练集 m 类同步抽审 ③评测增设\"可接受集合\"口径")


if __name__ == "__main__":
    main()
