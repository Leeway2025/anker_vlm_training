"""分类指标评测(与客户口径对齐: RT / SubKS 准确率 + KS 父类 + 安全关键召回)。

  python jax_impl/eval_metrics.py --preds preds.jsonl --labels labels.jsonl

preds 行: {"video_id", "output": "A | a | desc..."}(torch/JAX 推理同款)。
输出: RT acc / SubKS acc / 双对 acc / KS 父类 acc / 安全关键 SubKS 召回 /
格式合规率(可解析比例)。零 torch 依赖。
"""
import argparse
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.taxonomy import KS_GROUP, SAFETY_SK  # noqa: E402


def parse_output(text):
    m = re.match(r"\s*([A-E])\s*\|\s*([a-u])\s*\|", text or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def pct(x, d):
    return f"{100.0 * x / max(d, 1):.2f}%"


def compute(labels_path, preds, exclude=None, per_class=False):
    """统计一遍指标;exclude=需跳过的 video_id 集合(去噪口径用)。返回打印用 dict。"""
    exclude = exclude or set()
    n = rt_ok = sk_ok = both_ok = ks_ok = fmt_ok = miss = 0
    saf_tp = saf_n = skipped = 0
    per_sk = {}
    for l in open(labels_path, encoding="utf-8"):
        j = json.loads(l)
        vid = j["video_id"]
        if vid in exclude:
            skipped += 1
            continue
        rt, sk = j["labels"]["role_type"], j["labels"]["sub_keyscene"]
        n += 1
        # 口径统一(v1.8 修复): 缺失预测在所有指标中一律记错 ——
        # 旧版 continue 跳过了安全召回/每类统计的分母,分片推理只完成
        # 一部分时 acc 偏低而安全召回虚高,两个指标互相矛盾
        if vid not in preds:
            miss += 1
            prt = psk = None
        else:
            prt, psk = parse_output(preds[vid])
            if prt is not None:
                fmt_ok += 1
        rt_ok += (prt == rt)
        sk_ok += (psk == sk)
        both_ok += (prt == rt and psk == sk)
        ks_ok += (psk is not None and sk in KS_GROUP and psk in KS_GROUP
                  and KS_GROUP[psk] == KS_GROUP[sk])
        if sk in SAFETY_SK:
            saf_n += 1
            saf_tp += (psk == sk)
        st = per_sk.setdefault(sk, [0, 0])
        st[0] += 1
        st[1] += (psk == sk)
    return dict(n=n, miss=miss, fmt_ok=fmt_ok, rt_ok=rt_ok, sk_ok=sk_ok,
                both_ok=both_ok, ks_ok=ks_ok, saf_tp=saf_tp, saf_n=saf_n,
                skipped=skipped, per_sk=per_sk)


def report(r, per_class=False):
    print(f"samples={r['n']} missing_pred={r['miss']} "
          f"格式合规={pct(r['fmt_ok'], r['n'] - r['miss'])}")
    if r["miss"]:
        print(f"⚠️ {r['miss']} 条无预测,已在全部指标中记错(含安全召回);"
              f"若为分片未跑完,请先补齐再下结论")
    print(f"RoleType acc   = {pct(r['rt_ok'], r['n'])}")
    print(f"SubKS    acc   = {pct(r['sk_ok'], r['n'])}")
    print(f"RT+SubKS acc   = {pct(r['both_ok'], r['n'])}")
    print(f"KS 父类  acc   = {pct(r['ks_ok'], r['n'])}")
    print(f"安全关键 SubKS({SAFETY_SK}) 召回 = {pct(r['saf_tp'], r['saf_n'])} "
          f"(n={r['saf_n']})")
    if per_class:
        for sk in sorted(r["per_sk"]):
            c, k = r["per_sk"][sk]
            print(f"  [{sk}] n={c} acc={pct(k, c)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--exclude-ids", default="",
                    help="去噪口径: 该文件里的 video_id(每行一个)从测试集剔除后"
                         "【额外】再报一遍;原全量口径始终保留、不替换")
    a = ap.parse_args()

    preds = {j["video_id"]: j["output"] for j in
             (json.loads(l) for l in open(a.preds, encoding="utf-8"))}

    # ① 官方口径(全量)—— 主指标, 永远第一、永不改动
    print("== 官方口径(全量 test)==")
    report(compute(a.labels, preds, per_class=a.per_class), a.per_class)

    # ② 去噪口径(剔除 GT 自相矛盾且盲判证实的明显错标)—— 仅【附加】诊断
    if a.exclude_ids and os.path.exists(a.exclude_ids):
        excl = {x.strip() for x in open(a.exclude_ids, encoding="utf-8")
                if x.strip()}
        print(f"\n== 去噪口径(额外, 剔除 {len(excl)} 条明显错标 test)==")
        report(compute(a.labels, preds, exclude=excl, per_class=a.per_class),
               a.per_class)


if __name__ == "__main__":
    main()
