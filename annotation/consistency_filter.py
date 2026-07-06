"""标注过滤(torch-free)。两种模式:

  gt 模式(客户真实数据): Gemini 预测 vs 人工 GT,双分类字段全对 → 白名单
  double 模式(代理数据集): 两次独立 Gemini 采样,RT+SubKS 双次一致 → 伪 GT

用法:
  python -m annotation.consistency_filter --mode gt \
      --gemini pass1.jsonl --gt labels.jsonl --out-dir filtered/
  python -m annotation.consistency_filter --mode double \
      --gemini pass1.jsonl --gemini2 pass2.jsonl --out-dir filtered/
"""
import json
import argparse
import os
from collections import Counter


def _load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        out[d["video_id"]] = d
    return out


def _pred(d):
    p = d.get("gemini_output", d).get("predictions", {})
    return p.get("role_type"), p.get("sub_keyscene")


def filter_gt(gemini: dict, gts: dict):
    """客户数据: 与人工 GT 比对。产出白名单 + 部分匹配 + 疑似 GT 错标线索。"""
    white, partial, discard = [], [], []
    for vid, d in gemini.items():
        if vid not in gts:
            continue
        lab = gts[vid].get("labels", gts[vid])
        g_rt, g_sk = lab["role_type"], lab["sub_keyscene"]
        p_rt, p_sk = _pred(d)
        rt_m, sk_m = (p_rt == g_rt), (p_sk == g_sk)
        rec = {**d, "gt_rt": g_rt, "gt_sk": g_sk,
               "rt_match": rt_m, "sk_match": sk_m}
        if rt_m and sk_m:
            white.append(rec)
        elif rt_m or sk_m:
            partial.append(rec)
        else:
            discard.append(rec)   # 全错 → 抽查是否 GT 本身错标
    return white, partial, discard


def filter_double(pass1: dict, pass2: dict):
    """代理数据: 双次一致 → 伪 GT。"""
    white, discard = [], []
    for vid, d1 in pass1.items():
        d2 = pass2.get(vid)
        if d2 is None:
            continue
        rt1, sk1 = _pred(d1)
        rt2, sk2 = _pred(d2)
        if rt1 == rt2 and sk1 == sk2 and rt1 and sk1:
            # 伪 GT 采用 pass1(temperature 更低的一次)
            white.append({**d1, "pseudo_gt": {"role_type": rt1,
                                              "sub_keyscene": sk1}})
        else:
            discard.append({"video_id": vid,
                            "pass1": [rt1, sk1], "pass2": [rt2, sk2]})
    return white, discard


def _dump(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gt", "double"], required=True)
    ap.add_argument("--gemini", required=True)
    ap.add_argument("--gemini2", default=None)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    g1 = _load(a.gemini)
    if a.mode == "gt":
        assert a.gt, "--gt required in gt mode"
        white, partial, discard = filter_gt(g1, _load(a.gt))
        _dump(partial, os.path.join(a.out_dir, "partial_match.jsonl"))
        _dump(discard, os.path.join(a.out_dir, "discarded.jsonl"))
        # 疑似 GT 错标: Gemini 双字段全错的桶按类别统计,供人工抽查
        sus = Counter((d["gt_rt"], d["gt_sk"]) for d in discard)
        _dump([{"gt_combo": list(k), "count": v}
               for k, v in sus.most_common()],
              os.path.join(a.out_dir, "gt_suspect_stats.jsonl"))
    else:
        assert a.gemini2, "--gemini2 required in double mode"
        white, discard = filter_double(g1, _load(a.gemini2))
        _dump(discard, os.path.join(a.out_dir, "inconsistent.jsonl"))

    _dump(white, os.path.join(a.out_dir, "whitelist.jsonl"))
    with open(os.path.join(a.out_dir, "whitelist_ids.txt"), "w") as f:
        for d in white:
            f.write(d["video_id"] + "\n")
    total = len(g1)
    print(f"total={total} whitelist={len(white)} "
          f"rate={len(white)/total:.3f}" if total else "empty input")


if __name__ == "__main__":
    main()
