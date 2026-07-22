"""labels.jsonl 去重(每 video_id 保留首条)+ 前后分布对照。

  python3 jax_impl/dedup_labels.py --labels DATA/labels.jsonl \
      --out DATA/labels_dedup.jsonl

用途: 修复"复制凑配额"型注水(如 r 类 79 视频复制成 1580 行 —— 等效
把这 79 个视频的学习率放大 20 倍,且过报该类)。注意: 去重不改变
抽样配比 —— 自然先验请用全量池 dump_priors。纯 stdlib。
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seen = set()
    before = collections.Counter()
    after = collections.Counter()
    n_in = n_out = 0
    with open(a.out, "w", encoding="utf-8") as fout:
        for l in open(a.labels, encoding="utf-8"):
            j = json.loads(l)
            sk = j["labels"]["sub_keyscene"]
            before[sk] += 1
            n_in += 1
            if j["video_id"] in seen:
                continue
            seen.add(j["video_id"])
            after[sk] += 1
            n_out += 1
            fout.write(l)
    print(f"总行数 {n_in} → {n_out}(去掉 {n_in-n_out} 条重复,"
          f"{100.0*(n_in-n_out)/max(n_in,1):.1f}%)\n")
    print(f"{'类':<4}{'去重前':>8}{'去重后':>8}{'删除':>7}{'去重后占比':>11}")
    for sk in sorted(before, key=lambda k: -(before[k] - after[k])):
        cut = before[sk] - after[sk]
        flag = "  ⚠️ 复制重灾区" if cut > after[sk] else ""
        print(f"{sk:<4}{before[sk]:>8}{after[sk]:>8}{cut:>7}"
              f"{after[sk]/max(n_out,1):>10.1%}{flag}")
    print(f"\n[OK] -> {a.out}(v4 训练建议用此文件;自然先验仍取全量池)")


if __name__ == "__main__":
    main()
