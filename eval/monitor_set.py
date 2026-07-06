"""分层监控集构造(training_plan 2.1 ②;torch-free)。

随机 30k 里安全关键类可能只有几十条,监控曲线全是噪声;
本脚本从 labels.jsonl 构造 ~5k 分层集:
  - 安全关键类 q/r/u/n/j 各至多 N_SAFETY(默认 300)
  - 热区边界对 h/n、k/l、C/D 各至多 N_PAIR(默认 500)
  - 其余按自然分布补齐到 target_size

用法: python -m eval.monitor_set --labels labels.jsonl --out monitor.jsonl
"""
import json, argparse, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.taxonomy import SAFETY_SK  # noqa: E402

HOTSPOT_SK = ["h", "n", "k", "l"]
HOTSPOT_RT = ["C", "D"]


def build(records, target_size=5000, n_safety=300, n_pair=500, seed=0):
    rng = random.Random(seed)
    rng.shuffle(records)
    picked, picked_ids = [], set()

    def take(pred, cap):
        got = 0
        for r in records:
            if got >= cap:
                break
            lab = r.get("labels", r)
            if r["video_id"] not in picked_ids and pred(lab):
                picked.append(r); picked_ids.add(r["video_id"]); got += 1
        return got

    stats = {}
    for c in SAFETY_SK:
        stats[f"safety_{c}"] = take(lambda l, c=c: l["sub_keyscene"] == c, n_safety)
    for c in HOTSPOT_SK:
        stats[f"pair_sk_{c}"] = take(lambda l, c=c: l["sub_keyscene"] == c, n_pair)
    for c in HOTSPOT_RT:
        stats[f"pair_rt_{c}"] = take(lambda l, c=c: l["role_type"] == c, n_pair)
    stats["fill"] = take(lambda l: True, max(0, target_size - len(picked)))
    return picked, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=5000)
    a = ap.parse_args()
    records = [json.loads(l) for l in open(a.labels, encoding="utf-8")]
    picked, stats = build(records, a.size)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2), f"\ntotal={len(picked)}")


if __name__ == "__main__":
    main()
