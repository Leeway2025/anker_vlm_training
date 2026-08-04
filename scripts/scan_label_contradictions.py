#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全测试集"描述↔RoleType 类别自相矛盾"筛查(不止 A↔D 错例,含模型答对但标签本身错的)。
体系: A=家人 B=Staff(雇工/快递) C=可疑 D=未指定 E=非人。
只用【判别性】线索,门铃摄像头 residence/house 不作线索。启发式 review 清单,人工终判。纯 CPU。
检测项(每项都尽量选"硬"矛盾):
  [E1] 标E(非人) 但描述是人在做动作、无动物词        → 疑应人类类(A/B/C/D)
  [E2] 标非E 但描述纯动物主体、无人               → 疑应E
  [B1] 标B(Staff) 但描述含住户行为(钥匙/开锁自家门) → 疑应A
  [A1] 标A(家人) 但描述含 Staff/快递线索           → 疑应B
  [D1] 标D(未指定) 但描述含强住户线索(钥匙/开锁/门禁)→ 疑应A
用法: python3 scripts/scan_label_contradictions.py [--labels /data/labels_test.jsonl]
"""
import argparse, json, os, re
from collections import Counter

HUMAN = re.compile(r"\b(man|men|woman|women|person|people|child|children|boy|girl|kid|kids|guy|lady|male|female|someone|worker|courier|rider|teenager|elderly)\b", re.I)
ANIMAL = re.compile(r"\b(dogs?|cats?|raccoons?|deer|squirrels?|fox(es)?|coyotes?|birds?|animals?|wildlife|cows?|horses?|bears?|rabbits?|rats?|snakes?|possums?|skunks?|chickens?|goats?|pigs?|turkeys?)\b", re.I)
# 住户强线索(→A): 用钥匙/开锁自家门/门禁卡/自家
RES_STRONG = [
    (r"\bunlock", "unlock"),
    (r"\bkeys?\b", "key"),
    (r"\block(s|ed|ing)?\b[^.]{0,18}\bdoor", "lock the door"),
    (r"\baccess card\b", "access card"),
    (r"\b(his|her) own\b", "his/her own"),
]
# Staff/快递线索(→B)
STAFF = [
    (r"\b(delivery|courier|delivers?|delivered|delivering)\b", "delivery/courier"),
    (r"\b(package|parcel)s?\b", "package/parcel"),
    (r"\b(flyer|leaflet|pamphlet)s?\b", "flyer"),
    (r"\buniform\b", "uniform"),
    (r"\bworker\b", "worker"),
]


def match(text, table):
    t = (text or "").lower()
    for pat, name in table:
        if re.search(pat, t):
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/data/labels_test.jsonl")
    ap.add_argument("--out", default="outputs/diag/label_contradictions.jsonl")
    ap.add_argument("--sample", type=int, default=6)
    args = ap.parse_args()

    rows = []
    dist = Counter()
    for l in open(args.labels, encoding="utf-8"):
        d = json.loads(l)
        lb = d.get("labels") or d
        rt = lb.get("role_type"); desc = lb.get("description") or ""
        dist[rt] += 1
        has_h = bool(HUMAN.search(desc)); has_a = bool(ANIMAL.search(desc))
        res = match(desc, RES_STRONG); staff = match(desc, STAFF)
        flag = should = reason = cue = None
        if rt == "E":
            if has_h and not has_a:
                flag, should, cue = "E1", "人类(A/B/C/D)", "人在动作/无动物"
                reason = "标E(非人),但描述是人在做动作且无动物词"
            elif has_h and has_a:
                flag, should, cue = "E1s", "存疑", "人+动物并存"
                reason = "标E(非人),描述人与动物并存(主体可能是人)"
        else:
            if has_a and not has_h:
                flag, should, cue = "E2", "E(非人)", "纯动物主体"
                reason = f"标{rt},但描述是纯动物主体、无人"
            elif rt == "B" and res:
                flag, should, cue = "B1", "A(家人)", res
                reason = f"标B(Staff),但描述含住户行为『{res}』"
            elif rt == "A" and staff:
                flag, should, cue = "A1", "B(Staff)", staff
                reason = f"标A(家人),但描述含Staff/快递线索『{staff}』"
            elif rt == "D" and res:
                flag, should, cue = "D1", "A(家人)", res
                reason = f"标D(未指定),但描述含强住户线索『{res}』"
        if flag:
            rows.append({"video_id": d["video_id"], "gt_rt": rt, "flag": flag,
                         "suspect_should_be": should, "cue": cue, "reason": reason,
                         "gt_sk": lb.get("sub_keyscene"), "desc": desc})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    order = {"E1": 0, "E2": 1, "B1": 2, "A1": 3, "D1": 4, "E1s": 5}
    rows.sort(key=lambda r: order.get(r["flag"], 9))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fc = Counter(r["flag"] for r in rows)
    print(f"[in] {sum(dist.values())} 条 | RT分布 {dict(sorted(dist.items()))}")
    print(f"[out] {len(rows)} 条疑矛盾 -> {args.out}\n")
    names = {"E1": "标E但描述是人(硬)", "E2": "标非E但纯动物(疑应E)",
             "B1": "标B但住户行为(疑应A)", "A1": "标A但Staff/快递(疑应B)",
             "D1": "标D但强住户线索(疑应A)", "E1s": "标E人+动物并存(软/存疑)"}
    for k in ["E1", "E2", "B1", "A1", "D1", "E1s"]:
        base = dist.get(k[0] if k[0] in "ABCDE" else "E", 0)
        print(f"  [{k}] {names[k]}: {fc.get(k,0)}")
    for k in ["E1", "E2", "B1", "A1", "D1"]:
        sub = [r for r in rows if r["flag"] == k][:args.sample]
        if not sub:
            continue
        print(f"\n===== [{k}] {names[k]} 抽样(共{fc.get(k,0)}) =====")
        for r in sub:
            print(f"[{r['gt_rt']}→疑{r['suspect_should_be']}] sk={r['gt_sk']} cue『{r['cue']}』")
            print(f"   {r['desc'][:170]}")


if __name__ == "__main__":
    main()
