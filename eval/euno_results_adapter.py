"""euno 推理结果 json → eval/metrics 输入(基线复现同口径)。

euno 推理结果格式(euno数据集说明.md §推理结果格式):
  [{"id", "video", "conversations": [.., {"from":"gpt","value": GT}],
    "pred": {"result": "RT|SK|desc", "score": [..]}}, ...]

用法(客户复现基线 93.60/76.14/86.22 的完整命令):
  python -m eval.euno_results_adapter --results euno_infer_results.json \
      --out-dir baseline_eval/
  python -m eval.metrics --pred baseline_eval/preds.jsonl \
      --gt baseline_eval/gt.jsonl --out baseline_eval/report.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.euno_wds import parse_gpt_label      # noqa: E402


def convert(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_bad = 0
    with open(os.path.join(out_dir, "preds.jsonl"), "w") as fp, \
            open(os.path.join(out_dir, "gt.jsonl"), "w") as fg:
        for r in results:
            vid = r.get("video") or r.get("videos", [r.get("id")])[0]
            try:
                gt_value = next(c["value"] for c in r["conversations"]
                                if c["from"] == "gpt")
                rt, sk, desc = parse_gpt_label(gt_value)
            except (StopIteration, ValueError, KeyError):
                n_bad += 1
                continue
            fp.write(json.dumps({
                "video_id": vid, "output": r["pred"]["result"],
                "score": r["pred"].get("score")}) + "\n")
            fg.write(json.dumps({
                "video_id": vid,
                "labels": {"role_type": rt, "sub_keyscene": sk,
                           "description": desc}}) + "\n")
    return n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    results = json.load(open(a.results, encoding="utf-8"))
    n_bad = convert(results, a.out_dir)
    print(f"{len(results) - n_bad} pairs -> {a.out_dir}/(preds|gt).jsonl "
          f"(skipped {n_bad});接着跑 eval/metrics 即为客户口径报告")


if __name__ == "__main__":
    main()
