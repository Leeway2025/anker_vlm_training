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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--per-class", action="store_true")
    a = ap.parse_args()

    preds = {j["video_id"]: j["output"] for j in
             (json.loads(l) for l in open(a.preds, encoding="utf-8"))}
    n = rt_ok = sk_ok = both_ok = ks_ok = fmt_ok = miss = 0
    saf_tp = saf_n = 0
    per_sk = {}
    for l in open(a.labels, encoding="utf-8"):
        j = json.loads(l)
        vid = j["video_id"]
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

    def pct(x, d):
        return f"{100.0 * x / max(d, 1):.2f}%"

    print(f"samples={n} missing_pred={miss} 格式合规={pct(fmt_ok, n - miss)}")
    if miss:
        print(f"⚠️ {miss} 条无预测,已在全部指标中记错(含安全召回);"
              f"若为分片未跑完,请先补齐再下结论")
    print(f"RoleType acc   = {pct(rt_ok, n)}")
    print(f"SubKS    acc   = {pct(sk_ok, n)}")
    print(f"RT+SubKS acc   = {pct(both_ok, n)}")
    print(f"KS 父类  acc   = {pct(ks_ok, n)}")
    print(f"安全关键 SubKS({SAFETY_SK}) 召回 = {pct(saf_tp, saf_n)} "
          f"(n={saf_n})")
    if a.per_class:
        for sk in sorted(per_sk):
            c, k = per_sk[sk]
            print(f"  [{sk}] n={c} acc={pct(k, c)}")


if __name__ == "__main__":
    main()
