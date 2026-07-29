"""导出训练时的 val 切分为独立 jsonl(确定性复现,用于"考 val 卷子")。

  python3 jax_impl/export_val_split.py --labels DATA/labels.jsonl \
      --val-n 515 --seed 0 --out DATA/labels_val515.jsonl

参数必须与训练命令一致(--val-n 用训练日志 [data] 行里的实际 val 数,
--seed 同训练 --seed,默认 0)——split_by_camera 是确定性的,同参即同集。
用途: 对这份文件跑 infer_sharded + eval_metrics,若 val 上指标高而
测试集低 → 测试集标签口径/来源与训练集不一致;若 val 也低 → 训练
信号问题。纯 stdlib+本仓库,宿主机 python3 直接跑。
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_impl.data import split_by_camera  # noqa: E402


def match_mix_split(recs, mix_file, n_target, seed, fill_loose=False):
    """按参考文件的类配比切 val,camera 整组、缺口驱动贪心装箱。
    每轮给每个未选机位打分: score = Σ min(机位类数, 剩余缺口)
    − Σ 超额部分;收下正分最高者,循环至达标或无正分机位。"""
    sk = lambda r: (r.get("labels") or r)["sub_keyscene"]
    ref = [json.loads(l) for l in open(mix_file, encoding="utf-8")]
    ref_c = collections.Counter(sk(r) for r in ref)
    ref_n = sum(ref_c.values())
    avail = collections.Counter(sk(r) for r in recs)
    quota = {k: min(round(v / ref_n * n_target), avail.get(k, 0))
             for k, v in ref_c.items()}
    by_cam = collections.defaultdict(list)
    for r in recs:
        cam = (r.get("meta") or {}).get("camera_id") or r["video_id"]
        if cam == "unknown":
            cam = r["video_id"]
        by_cam[cam].append(r)
    cam_cnt = {c: collections.Counter(sk(r) for r in g)
               for c, g in by_cam.items()}
    cams = sorted(by_cam)
    random.Random(seed).shuffle(cams)
    got = collections.Counter()
    chosen = []
    remaining = set(cams)
    while sum(got.values()) < n_target:
        best, best_s = None, 0.0
        for c in remaining:
            s = 0.0
            for k, v in cam_cnt[c].items():
                deficit = max(0, quota.get(k, 0) - got[k])
                s += min(v, deficit) - max(0, v - deficit)
            if s > best_s:
                best, best_s = c, s
        if best is None:
            break
        chosen.append(best)
        remaining.discard(best)
        got.update(cam_cnt[best])
    val = [r for c in chosen for r in by_cam[c]]
    if fill_loose:
        in_val = {r["video_id"] for r in val}
        pool = collections.defaultdict(list)
        for c in remaining:
            for r in by_cam[c]:
                pool[sk(r)].append(r)
        rng2 = random.Random(seed + 1)
        n_fill = 0
        for k, q in quota.items():
            need = q - got[k]
            if need > 0 and pool.get(k):
                rng2.shuffle(pool[k])
                take = pool[k][:need]
                val += take
                got.update({k: len(take)})
                n_fill += len(take)
        if n_fill:
            print(f"[match-mix] ⚠️ 松散补齐 {n_fill} 条(该部分机位与 train "
                  f"重叠,相关类的 val 读数会略乐观)")
    print(f"[match-mix] 目标 {n_target} → 实切 {len(val)};配比对照:")
    for k in sorted(quota, key=lambda k: -quota[k]):
        if quota[k]:
            print(f"  [{k}] 配额 {quota[k]:>4} 实得 {got[k]:>4}")
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="训练用 labels.jsonl")
    ap.add_argument("--val-n", type=int, required=True,
                    help="目标 val 条数(match-mix 模式为近似目标)")
    ap.add_argument("--match-mix", default=None,
                    help="传参考 jsonl(如 labels_test): val 类配比对齐它;"
                         "camera 整组贪心装箱")
    ap.add_argument("--fill-loose", action="store_true",
                    help="整机位装箱后仍有缺口的类,按单条补齐(牺牲该部分"
                         "机位完整性 → 这些类的 val 读数会略乐观,会告警)")
    ap.add_argument("--seed", type=int, default=0, help="同训练 --seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-out", default=None,
                    help="同时导出 video_id 清单(供 train_sft --val-ids)")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.labels, encoding="utf-8")]
    if a.match_mix:
        va = match_mix_split(recs, a.match_mix, a.val_n, a.seed,
                             fill_loose=a.fill_loose)
    else:
        _, va = split_by_camera(recs, a.val_n, seed=a.seed)
    if a.ids_out:
        with open(a.ids_out, "w") as f:
            for r in va:
                f.write(r["video_id"] + "\n")
        print(f"[OK] val_ids({len(va)} 条)-> {a.ids_out}")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in va:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] val {len(va)} 条 -> {a.out}"
          + (f"(⚠️ 与 --val-n {a.val_n} 不等,请核对 seed/labels 是否与训练一致)"
             if len(va) != a.val_n else ""))


if __name__ == "__main__":
    main()
