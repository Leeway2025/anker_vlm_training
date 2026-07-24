"""v5-data 清洗轮数据构造: 按三方对质 D 桶剔除错标重嫌,产出新训练集。

  python3 jax_impl/build_v5_data.py --labels /data/labels_dedup.jsonl \
      --clean-ids outputs/cross_exam/clean_candidates.txt \
      --out labels_v5.jsonl

行为:
  - 剔除 clean-ids 内的行(训练器 sample-weights 只能上权 ≥1.0,
    降权不可用,清洗唯一路径 = 物理剔除);
  - 双保险: GT=m 的行即使出现在清单里也不剔(冻结基准下 m 懒标与
    测试集同源,像基准比对基准重要 —— cross_examine 已排,这里再守一道);
  - 安全护栏: 任何类剔除比例 >20% 即拒绝执行(错标不该这么密,
    大概率是清单来源出了问题,先人工抽查);
  - 打印前后类分布对照(剔除会移动先验,推理侧先验校正需重标定)。
纯 stdlib。产物直接替换 train_sft --labels(单变量 = 数据版本)。
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--clean-ids", required=True)
    ap.add_argument("--max-class-frac", type=float, default=0.20,
                    help="单类剔除比例熔断线")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids = set(open(a.clean_ids).read().split())
    rows = [json.loads(l) for l in open(a.labels, encoding="utf-8")]
    sk = lambda r: (r.get("labels") or r)["sub_keyscene"]

    before = collections.Counter(sk(r) for r in rows)
    drop = [r for r in rows
            if r["video_id"] in ids and sk(r) != "m"]     # m 双保险
    m_shielded = sum(1 for r in rows
                     if r["video_id"] in ids and sk(r) == "m")
    drop_c = collections.Counter(sk(r) for r in drop)
    for k, n in drop_c.items():
        frac = n / max(before[k], 1)
        if frac > a.max_class_frac:
            raise SystemExit(
                f"❌ 熔断: 类 [{k}] 将剔除 {n}/{before[k]} ({frac:.0%}) "
                f"> {a.max_class_frac:.0%} —— 错标不应这么密,先人工抽查"
                f"清单来源(mislabel_suspects.jsonl 抽 20 条看帧)")

    drop_ids = {r["video_id"] for r in drop}
    kept = [r for r in rows if r["video_id"] not in drop_ids]
    with open(a.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    after = collections.Counter(sk(r) for r in kept)
    print(f"{'类':<4}{'前':>7}{'剔':>6}{'后':>7}{'占比变化':>12}")
    nb, na = sum(before.values()), sum(after.values())
    for k in sorted(before, key=lambda k: -drop_c.get(k, 0)):
        d = after[k] / na - before[k] / nb
        print(f"{k:<4}{before[k]:>7}{drop_c.get(k, 0):>6}{after[k]:>7}"
              f"{d:>+11.2%}")
    print(f"\n共剔 {len(drop_ids)} 行(m 类清单命中 {m_shielded} 条已豁免)"
          f";{nb} -> {na} 行 -> {a.out}")
    print("提醒: ①重训命令 = v4 配方仅换 --labels(单变量);"
          "②类先验已移动,评测时推理侧先验校正的 Δ 需重新 dump 标定")


if __name__ == "__main__":
    main()
