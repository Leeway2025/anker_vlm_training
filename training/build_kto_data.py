"""KTO 偏好数据构造(training_plan 10.1: 纯分类偏好,不跑 judge)。

  desirable   = (video, GT 完整输出串)          全部白名单样本
  undesirable = (video, 模型的错误输出串)        仅分类错误样本

纯逻辑(build_pairs)可单测。用法:
  python -m training.build_kto_data --preds preds.jsonl \
      --labels labels.jsonl --whitelist wl.txt --out kto_data.jsonl
"""
import json, argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.format_validator import parse_output       # noqa: E402
from data.formatting import build_target             # noqa: E402


def build_pairs(preds: dict, gt_records: dict, sep="|"):
    """gt_records: video_id → labels dict。返回 KTO 样本列表。"""
    out = []
    for vid, lab in gt_records.items():
        gt_text = build_target(lab["role_type"], lab["sub_keyscene"],
                               lab.get("description", ""), sep).text
        out.append({"video_id": vid, "completion": gt_text, "label": True})
        pred = preds.get(vid)
        if pred is None:
            continue
        r = parse_output(pred)
        wrong = (not r.ok) or r.rt != lab["role_type"] \
            or r.sk != lab["sub_keyscene"]
        if wrong:
            out.append({"video_id": vid, "completion": pred.strip(),
                        "label": False})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--whitelist", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wl = set(open(a.whitelist).read().split()) if a.whitelist else None
    preds = {json.loads(l)["video_id"]: json.loads(l)["output"]
             for l in open(a.preds, encoding="utf-8")}
    gts = {}
    for l in open(a.labels, encoding="utf-8"):
        d = json.loads(l)
        if wl and d["video_id"] not in wl:
            continue
        gts[d["video_id"]] = d.get("labels", d)
    # separator 以 base.yaml 为单一来源(euno 真实 GT 已核对为 "|",
    # desirable 串格式错误会让 KTO 直接教坏模型)
    import os
    sep = "|"
    if os.path.exists("configs/base.yaml"):
        import yaml
        sep = yaml.safe_load(open("configs/base.yaml",
                                  encoding="utf-8"))["format"]["separator"]
    pairs = build_pairs(preds, gts, sep)
    with open(a.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_u = sum(1 for p in pairs if not p["label"])
    print(f"total={len(pairs)} desirable={len(pairs)-n_u} undesirable={n_u}")


if __name__ == "__main__":
    main()
