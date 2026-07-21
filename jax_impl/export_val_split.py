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
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_impl.data import split_by_camera  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="训练用 labels.jsonl")
    ap.add_argument("--val-n", type=int, required=True,
                    help="训练日志 [data] 行的 val 数(如 515)")
    ap.add_argument("--seed", type=int, default=0, help="同训练 --seed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.labels, encoding="utf-8")]
    _, va = split_by_camera(recs, a.val_n, seed=a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in va:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] val {len(va)} 条 -> {a.out}"
          + (f"(⚠️ 与 --val-n {a.val_n} 不等,请核对 seed/labels 是否与训练一致)"
             if len(va) != a.val_n else ""))


if __name__ == "__main__":
    main()
