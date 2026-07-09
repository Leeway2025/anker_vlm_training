"""Hard Example Mining(training_plan 6.4:按难度上采样,保持类别分布)。

流程: checkpoint 推理训练集 → parse 对比 GT → 错例上采样 → 续训。
纯逻辑(build_weights)可单测;错例判定只看分类字段(分类优先)。

⚠️ 朴素"错例 ×3"的三个下滑风险与对应防护(2026-07-08 加固):
  ① 错例扎堆在难类(h/n、k/l、s、C/D)→ 全部 ×3 会隐性改写类别先验,
    模型过度预测难类、常见类被挤压,自然分布测试集上总 ACC 可能净降
    → **per-class 质量膨胀上限**(--max-class-inflation,默认 1.5×):
      某类错例过多时自动压低该类的 hard 权重,类别先验近似保持
  ② "模型错了"里混着"GT 错了"(h/n 边界噪声预期 ~15%),放大 3 倍
    等于加倍教错标签 → **--whitelist(资产 D)**: 只放大 Gemini 与
    GT 一致的可信错例,疑似 GT 噪声保持 ×1 并单独计数上报
  ③ 续训分布被难例主导 → 易样本遗忘 → 防护在训练侧: 续训必须带
    验证集早停 + 监控集刹车(REPRODUCE 验收: 总指标不降才算过)

用法:
  python -m training.hard_mining --preds preds.jsonl --labels labels.jsonl \
      --out sample_weights.json --weight 3.0 \
      [--whitelist DATA/asset_D_whitelist.txt] [--max-class-inflation 1.5]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.format_validator import parse_output  # noqa: E402


def build_weights(preds: dict, gts: dict, hard_weight: float = 3.0,
                  whitelist=None, max_class_inflation: float = 1.5):
    """返回 (weights: video_id→w, stats)。

    gts: vid → (rt, sk)。类别键 = sk(场景先验是客户主指标的主轴)。
    per-class 上限: 类 c 有 n_c 条、可信错例 k_c 条,则质量膨胀
    (n_c + (w-1)·k_c)/n_c ≤ cap → w_c = min(w, 1 + (cap-1)·n_c/k_c)。
    """
    wl = set(whitelist) if whitelist else None
    per_cls = {}                       # sk → {"n": 总数, "hard": [vid...]}
    weights = {}
    stats = {"correct": 0, "wrong_trusted": 0, "wrong_untrusted": 0,
             "missing_pred": 0}
    for vid, (g_rt, g_sk) in gts.items():
        c = per_cls.setdefault(g_sk, {"n": 0, "hard": []})
        c["n"] += 1
        weights[vid] = 1.0
        if vid not in preds:
            stats["missing_pred"] += 1
            continue
        r = parse_output(preds[vid])
        wrong = (not r.ok) or r.rt != g_rt or r.sk != g_sk
        if not wrong:
            stats["correct"] += 1
        elif wl is not None and vid not in wl:
            stats["wrong_untrusted"] += 1     # 疑似 GT 噪声,不放大
        else:
            stats["wrong_trusted"] += 1
            c["hard"].append(vid)

    stats["class_inflation"] = {}
    for sk, c in per_cls.items():
        k = len(c["hard"])
        if not k:
            continue
        w = hard_weight
        if max_class_inflation:
            w = min(w, 1.0 + (max_class_inflation - 1.0) * c["n"] / k)
        for vid in c["hard"]:
            weights[vid] = w
        stats["class_inflation"][sk] = {
            "n": c["n"], "hard": k, "weight": round(w, 2),
            "mass_x": round((c["n"] + (w - 1) * k) / c["n"], 2)}
    return weights, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight", type=float, default=3.0)
    ap.add_argument("--whitelist", default=None,
                    help="资产 D: 只放大 Gemini 确认过 GT 的错例")
    ap.add_argument("--max-class-inflation", type=float, default=1.5,
                    help="每类训练质量膨胀上限(0=不限,复现旧行为)")
    a = ap.parse_args()
    preds = {json.loads(l)["video_id"]: json.loads(l)["output"]
             for l in open(a.preds, encoding="utf-8")}
    gts = {}
    for l in open(a.labels, encoding="utf-8"):
        d = json.loads(l)
        lab = d.get("labels", d)
        gts[d["video_id"]] = (lab["role_type"], lab["sub_keyscene"])
    wl = set(open(a.whitelist).read().split()) if a.whitelist else None
    weights, stats = build_weights(preds, gts, a.weight, wl,
                                   a.max_class_inflation)
    json.dump(weights, open(a.out, "w"))
    print(json.dumps(stats, ensure_ascii=False, indent=1), "->", a.out)


if __name__ == "__main__":
    main()
