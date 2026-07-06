"""Hard Example Mining(training_plan 6.4:按难度上采样,保持类别分布)。

流程: checkpoint 推理训练集 → parse 对比 GT → 错例 3x 权重
纯逻辑(build_weights)可单测;错例判定只看分类字段(分类优先)。

用法:
  python -m training.hard_mining --preds preds.jsonl --labels labels.jsonl \
      --out sample_weights.json --weight 3.0
  (preds 由 inference_utils.generate_predictions 产出)
"""
import json, argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.format_validator import parse_output  # noqa: E402


def build_weights(preds: dict, gts: dict, hard_weight: float = 3.0):
    """返回 (weights: video_id→w, stats)。错例(分类任一字段错/格式失败)= hard_weight。
    保持类别分布: 不按类别倒频率,只按对错。"""
    weights, stats = {}, {"correct": 0, "wrong": 0, "missing_pred": 0}
    for vid, (g_rt, g_sk) in gts.items():
        if vid not in preds:
            weights[vid] = 1.0
            stats["missing_pred"] += 1
            continue
        r = parse_output(preds[vid])
        wrong = (not r.ok) or r.rt != g_rt or r.sk != g_sk
        weights[vid] = hard_weight if wrong else 1.0
        stats["wrong" if wrong else "correct"] += 1
    return weights, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight", type=float, default=3.0)
    a = ap.parse_args()
    preds = {json.loads(l)["video_id"]: json.loads(l)["output"]
             for l in open(a.preds, encoding="utf-8")}
    gts = {}
    for l in open(a.labels, encoding="utf-8"):
        d = json.loads(l); lab = d.get("labels", d)
        gts[d["video_id"]] = (lab["role_type"], lab["sub_keyscene"])
    weights, stats = build_weights(preds, gts, a.weight)
    json.dump(weights, open(a.out, "w"))
    print(json.dumps(stats), "->", a.out)


if __name__ == "__main__":
    main()
