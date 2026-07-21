"""labels.jsonl 数据体检: 每类 行数 vs 独立视频数(查配平是否靠复制)。

  python3 jax_impl/label_stats.py --labels DATA/labels.jsonl

判读: 某类 行数≫独立视频数(dup 高)→ 配平靠复制行实现,模型会反复
背同一批视频,易形成 attractor 类(预测大量涌向该类);独立 camera 数
过少的类同理(多样性假象)。纯 stdlib。
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()
    rows = collections.Counter()
    vids = collections.defaultdict(set)
    cams = collections.defaultdict(set)
    dup_rows = 0
    seen = set()
    for l in open(a.labels, encoding="utf-8"):
        j = json.loads(l)
        sk = j["labels"]["sub_keyscene"]
        vid = j["video_id"]
        rows[sk] += 1
        vids[sk].add(vid)
        cams[sk].add((j.get("meta") or {}).get("camera_id") or vid)
        if vid in seen:
            dup_rows += 1
        seen.add(vid)
    total = sum(rows.values())
    print(f"总行数 {total}  独立视频 {len(seen)}  重复行 {dup_rows}"
          f"({100.0*dup_rows/max(total,1):.1f}%)")
    print(f"{'类':<3}{'行数':>8}{'独立视频':>10}{'dup倍率':>9}{'独立camera':>12}")
    for sk in sorted(rows, key=lambda k: -rows[k]):
        n, u, c = rows[sk], len(vids[sk]), len(cams[sk])
        flag = "  ⚠️ 复制配平" if n / max(u, 1) > 1.5 else ""
        print(f"{sk:<3}{n:>8}{u:>10}{n/max(u,1):>9.2f}{c:>12}{flag}")


if __name__ == "__main__":
    main()
