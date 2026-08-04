#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 A↔D 错例里筛"高嫌疑错标 / 不可判"清单 —— "更干净标注"的第一份弹药。
体系: A=家人 B=Staff(雇工/快递) C=可疑 D=未指定(=other/判不了) E=非人。
      关键: D 是"身份未指定"的兜底桶,不是"陌生/快递";快递员按体系是 B(Staff)。
判据分级(越低越硬):
  Tier1 = GT 描述与 GT 字母标签【自相矛盾】(不依赖信任模型):
          标D 却含住户线索(锁门/钥匙/进自家) → 疑应A;
          标A 却含 Staff/快递线索(送包裹/快递) → 疑应B(不是D!)。
  Tier2 = 较弱证据: 只有【访客线索】(门铃/敲门,只能证"疑非家人",D或B待定),
          或 GT 描述无线索但【模型描述】含线索且模型判向一致。
  Tier3 = GT 与模型描述都无身份判别线索(本质不可判)→ 建议降权/剔除,非错标。
注: 全库都是门铃摄像头,"residence/house/home" 不作线索;只用【判别性】线索词。
    v2 校正: 快递/包裹/传单 → 疑B(原误判为D); 删除 mailbox(取自家信=住户,非陌生);
             门铃/敲门 → 疑"非A"(D/B待定,原硬判D)。
    这是启发式 review 清单,人工/客户终判。纯 CPU。
用法: python3 scripts/flag_ad_mislabels.py [--in outputs/diag/ad_errors.jsonl]
"""
import argparse, json, os, re

# 住户强线索(→A): 锁/开自家门、用钥匙、进自己家、自家、家人、回家
RES = [
    (r"\block(s|ed|ing)?\b[^.]{0,18}\bdoor", "lock the door"),
    (r"\bunlock", "unlock"),
    (r"\bkeys?\b", "key"),
    (r"\benters?\b[^.]{0,15}\b(house|home|residence)", "enters house"),
    (r"\b(goes?|went|walk(s|ed)?)\b[^.]{0,10}\binside\b", "goes inside"),
    (r"\binto\b[^.]{0,10}\b(his|her|the)\b[^.]{0,6}\b(house|home)", "into the house"),
    (r"\b(his|her) own\b", "his/her own"),
    (r"\bfamily\b", "family"),
    (r"\breturns?\b[^.]{0,8}\bhome\b", "returns home"),
    (r"\baccess card\b", "access card"),
]
# Staff/雇工强线索(→B): 快递、包裹、传单、制服、工人 (快递员=B,非D)
STAFF = [
    (r"\b(delivery|courier|delivers?|delivered|delivering)\b", "delivery/courier"),
    (r"\b(package|parcel)s?\b", "package/parcel"),
    (r"\b(flyer|leaflet|pamphlet)s?\b", "flyer"),
    (r"\buniform\b", "uniform"),
    (r"\bworker\b", "worker"),
]
# 访客弱线索(→非A,D或B待定): 门铃、敲门 —— 只能证"疑非家人",分不出 D/B
VISITOR = [
    (r"\bdoorbell\b", "doorbell"),
    (r"\brang?\b[^.]{0,10}\b(bell|door)", "rang bell"),
    (r"\bpress(es|ed)?\b[^.]{0,10}\b(bell|doorbell|button)", "press doorbell"),
    (r"\bknock", "knock"),
]


def match(text, table):
    t = (text or "").lower()
    for pat, name in table:
        if re.search(pat, t):
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/diag/ad_errors.jsonl")
    ap.add_argument("--out", default="outputs/diag/ad_mislabel_suspects.jsonl")
    ap.add_argument("--sample", type=int, default=15)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    out = []
    for r in rows:
        gt = r["gt_rt"]                       # A 或 D(A↔D 错例)
        g_res = match(r["gt_desc"], RES); g_staff = match(r["gt_desc"], STAFF); g_vis = match(r["gt_desc"], VISITOR)
        p_res = match(r["pred_desc"], RES); p_staff = match(r["pred_desc"], STAFF); p_vis = match(r["pred_desc"], VISITOR)
        g_any = g_res or g_staff or g_vis
        p_any = p_res or p_staff or p_vis
        tier = should = reason = cue = None
        if gt == "D":                         # 标"未指定/other"
            if g_res and not (g_staff or g_vis):
                tier, should, cue = 1, "A", g_res
                reason = f"GT标D(未指定),但GT描述含住户线索『{g_res}』(自相矛盾,疑应A)"
            elif g_staff and not (g_res or g_vis):
                tier, should, cue = 2, "B", g_staff
                reason = f"GT标D(未指定),但GT描述含Staff线索『{g_staff}』(疑应更精确为B)"
            elif p_res and not g_any and not (p_staff or p_vis):
                tier, should, cue = 2, "A", p_res
                reason = f"GT描述无判别线索;模型描述含住户线索『{p_res}』且模型判A"
        elif gt == "A":                       # 标"家人"
            if g_staff and not g_res:
                tier, should, cue = 1, "B", g_staff
                reason = f"GT标A(家人),但GT描述含Staff/快递线索『{g_staff}』(自相矛盾,疑应B)"
            elif g_vis and not g_res:
                tier, should, cue = 2, "非A", g_vis
                reason = f"GT标A(家人),但GT描述含访客线索『{g_vis}』(疑非家人,D或B待定)"
            elif p_staff and not g_any and not p_vis:
                tier, should, cue = 2, "B", p_staff
                reason = f"GT描述无判别线索;模型描述含Staff线索『{p_staff}』(疑应B)"
            elif p_vis and not g_any and not p_staff:
                tier, should, cue = 2, "非A", p_vis
                reason = f"GT描述无判别线索;模型描述含访客线索『{p_vis}』(疑非家人,D/B待定)"
        if tier is None:                      # 两侧都无判别线索 → 本质不可判
            if not (g_any or p_any):
                tier, should, reason, cue = 3, "?", "GT与模型描述均无身份判别线索(本质不可判,建议降权/剔除)", ""
            else:
                continue                      # 线索混杂,暂不入清单
        out.append({"video_id": r["video_id"], "gt_rt": gt, "suspect_should_be": should,
                    "tier": tier, "reason": reason, "cue": cue,
                    "p_pred": r["p_pred"], "p_gt": r["p_gt"], "gt_sk": r["gt_sk"],
                    "gt_desc": r["gt_desc"], "pred_desc": r["pred_desc"]})

    out.sort(key=lambda x: (x["tier"], -x["p_pred"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Tier1 单独出一份纯 video_id 清单(交标注最省事)
    t1_ids = [r["video_id"] for r in out if r["tier"] == 1]
    with open("outputs/diag/ad_mislabel_tier1_ids.txt", "w") as f:
        f.write("\n".join(t1_ids) + ("\n" if t1_ids else ""))
    # 疑 B(Staff)漏标清单(跨 tier,快递/雇工被压进 A/D)—— 交标注单独一份
    b_ids = [r["video_id"] for r in out if r["suspect_should_be"] == "B"]
    with open("outputs/diag/staff_suspects_ids.txt", "w") as f:
        f.write("\n".join(b_ids) + ("\n" if b_ids else ""))

    from collections import Counter
    c = Counter(r["tier"] for r in out)
    sc = Counter(r["suspect_should_be"] for r in out)
    print(f"[out] {len(out)}/{len(rows)} 条入清单 -> {args.out}")
    print(f"      Tier1 自相矛盾(硬): {c[1]}  | Tier2 弱证据: {c[2]}  | Tier3 不可判: {c[3]}")
    print(f"      建议去向: 疑A {sc['A']} | 疑B(Staff) {sc['B']} | 疑非A(D/B待定) {sc['非A']} | 不可判 {sc['?']}")
    print(f"      Tier1 video_id 清单 -> outputs/diag/ad_mislabel_tier1_ids.txt ({len(t1_ids)} 条)")
    print(f"      疑B(Staff)漏标清单  -> outputs/diag/staff_suspects_ids.txt ({len(b_ids)} 条)")
    # Tier1 去向拆分
    t1 = [r for r in out if r["tier"] == 1]
    d2a = sum(1 for r in t1 if r["suspect_should_be"] == "A")
    a2b = sum(1 for r in t1 if r["suspect_should_be"] == "B")
    print(f"      Tier1 拆向: 标D疑应A {d2a} | 标A疑应B {a2b}")

    print(f"\n================ Tier1 硬嫌疑错标 全部(共{c[1]}) ================")
    for r in t1[:args.sample]:
        print(f"\n[{r['video_id']}]  标{r['gt_rt']}→疑应{r['suspect_should_be']}  cue『{r['cue']}』  p判错={r['p_pred']}")
        print(f"  GT 描述 : {r['gt_desc'][:190]}")
        print(f"  模型描述: {r['pred_desc'][:190]}")


if __name__ == "__main__":
    main()
