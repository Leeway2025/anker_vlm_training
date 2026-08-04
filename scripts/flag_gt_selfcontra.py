#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练集 RoleType 明显错标修正候选 —— 唯一正收益口径:【GT 描述自相矛盾】当闸,只升级,
   Gemini盲判 + 模型预测 只做旁证加信心。人工/客户终判后再进单变量重训。纯 CPU。
体系: A=家人 B=Staff(雇工/快递) C=可疑(明确犯罪可能性行为) D=未指定 E=非人(动物/车辆)。
四个升级方向(只升不降 —— 不做 C→D,因 Gemini 中性 captioner 读不出威胁,降级=错杀):
  →B: GT描述含 雇工/快递/制服/消防/维修 身份词, 但标 A/D
  →C: GT描述含 试门/撬门/破窗/翻墙/持械/闯入/盗窃 犯罪行为, 但标 A/B/D
  →E: GT描述纯车辆/纯动物、无人, 但标 A/B/C/D
  →A: GT描述含 用钥匙/门禁开【自家门】(排除车锁/车钥匙/自行车/后备箱/信箱), 但标 D
信心分级: strong=GT矛盾+模型与Gemini都改判同向; medium=GT矛盾+其一改判; weak=仅GT矛盾。
用法: python3 scripts/flag_gt_selfcontra.py --labels /data/labels_dedup.jsonl \
        --gemini <blind.jsonl> --preds <train_preds.jsonl> --out outputs/diag/gt_selfcontra_train.jsonl
"""
import argparse, json, os, re
from collections import Counter

# 雇工身份词(→B) —— 只用【人的身份】词, 不用裸 package(住户也拿自家快递)
STAFF = re.compile(r"\b(staff|service worker|worker|uniform|courier|delivery (person|man|driver|worker)|deliveryman|postal|mail ?man|mail carrier|ups|fedex|amazon|dhl|technician|maintenance|repairman|plumber|electrician|gardener|landscaper|cleaner|firefighter|police officer|officer|meter reader)\b", re.I)
# 明确犯罪可能性行为(→C)
CRIME = re.compile(r"\b(tries? to (open|force|pry)|tried to (open|force|pry)|attempts? to (open|force|enter)|forces? (the |open )?(door|window|lock)|pry(ing|s|ed)?\b|forced entry|breaks? in(to)?\b|broke (in|into|a window)|smash(es|ed|ing)?|climb(s|ed|ing)? (over |the )?(fence|wall|gate|window)|jump(s|ed)? (the |over )?(fence|wall)|weapon|gun|knife|firearm|pistol|rifle|steals?|stole|theft|robber|burglar|tamper|vandal|graffiti|kicks? (the )?door|pickpocket)\b", re.I)
HUMAN = re.compile(r"\b(man|men|woman|women|person|people|child|children|boy|girl|kid|kids|guy|lady|male|female|someone|worker|courier|rider|teenager|elderly|resident|individual|pedestrian)\b", re.I)
ANIMAL = re.compile(r"\b(dogs?|cats?|raccoons?|deer|squirrels?|fox(es)?|coyotes?|birds?|animals?|wildlife|cows?|horses?|bears?|rabbits?|rats?|possums?|skunks?)\b", re.I)
VEHICLE = re.compile(r"\b(car|cars|vehicle|truck|van|suv|sedan|pickup|motorcycle|scooter)\b", re.I)
VMOVE = re.compile(r"\b(driv(e|es|ing)|park(s|ed|ing)?|pass(es|ed|ing)?\b|reverses?|pulls? (in|out|up|away))\b", re.I)
# 自家门线索(→A) —— 钥匙/门禁 且靠近 门/屋, 且不在车/自行车语境
RESDOOR = re.compile(r"\b(unlock(s|ed|ing)?|uses? (a |the )?key|access card|key ?card|swipe[sd]?)\b[^.]{0,25}\b(door|house|home|residence|gate|apartment)", re.I)
RESLOCK = re.compile(r"\block(s|ed|ing)?\b[^.]{0,20}\b(house|home|residence|front) ?door", re.I)
CARCTX = re.compile(r"\b(car|vehicle|truck|van|suv|sedan|pickup|bike|bicycle|trunk|mailbox)\b", re.I)


def rt_of(d):
    lb = d.get("labels") or d
    return lb.get("role_type"), lb.get("sub_keyscene"), lb.get("description", "")


def load(p, kind):
    o = {}
    for l in open(p, encoding="utf-8"):
        d = json.loads(l); v = d["video_id"]
        if kind == "gt":
            o[v] = rt_of(d)
        elif kind == "gem":
            g = (d.get("gemini_output") or d).get("predictions") or {}
            o[v] = g.get("role_type")
        else:
            seg = (d.get("output") or "").split("|")
            o[v] = seg[0].strip() if seg else None
    return o


def suspect(gt_rt, desc):
    """返回 (should, direction_cue) 或 None —— GT 自相矛盾的升级方向。"""
    if gt_rt in ("A", "D") and STAFF.search(desc):
        return "B", "staff:" + STAFF.search(desc).group(0)
    if gt_rt in ("A", "B", "D") and CRIME.search(desc):
        return "C", "crime:" + CRIME.search(desc).group(0)
    if gt_rt in ("A", "B", "C", "D") and (ANIMAL.search(desc) or (VEHICLE.search(desc) and VMOVE.search(desc))) and not HUMAN.search(desc):
        return "E", "nonhuman:" + ("animal" if ANIMAL.search(desc) else "vehicle-only")
    if gt_rt == "D":
        m = RESDOOR.search(desc) or RESLOCK.search(desc)
        if m and not CARCTX.search(m.group(0)):
            return "A", "resdoor:" + m.group(0)[:30]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/data/labels_dedup.jsonl")
    ap.add_argument("--gemini", default="/home/nas-tpu-poc/data/zx_vlm_dataset/anker_video_clips/euno_train_v3.0.18_balanced_100k_labels.jsonl")
    ap.add_argument("--preds", default="outputs/kto_prep/train_preds.jsonl")
    ap.add_argument("--out", default="outputs/diag/gt_selfcontra_train.jsonl")
    ap.add_argument("--sample", type=int, default=10)
    a = ap.parse_args()

    gt = load(a.labels, "gt"); gm = load(a.gemini, "gem"); md = load(a.preds, "pred")
    rows = []
    for v, (grt, gsk, desc) in gt.items():
        s = suspect(grt, desc)
        if not s:
            continue
        should, cue = s
        m_rt = md.get(v); z_rt = gm.get(v)
        m_agree = (m_rt == should); z_agree = (z_rt == should)
        conf = "strong" if (m_agree and z_agree) else ("medium" if (m_agree or z_agree) else "weak")
        rows.append({"video_id": v, "gt_rt": grt, "suspect": should, "conf": conf,
                     "cue": cue, "model_rt": m_rt, "gem_rt": z_rt, "gt_sk": gsk, "desc": desc})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ORD = {"strong": 0, "medium": 1, "weak": 2}
    rows.sort(key=lambda r: (ORD[r["conf"]], r["suspect"]))
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dirc = Counter(f"{r['gt_rt']}→{r['suspect']}" for r in rows)
    cc = Counter(r["conf"] for r in rows)
    print(f"[in] 三方: gt={len(gt)} gem={len(gm)} pred={len(md)}")
    print(f"[out] {len(rows)} 条 GT自相矛盾升级候选 -> {a.out}")
    print(f"  信心: strong(模型+Gemini都同向) {cc['strong']} | medium(其一) {cc['medium']} | weak(仅GT矛盾) {cc['weak']}")
    print(f"  方向: {dict(dirc.most_common())}")
    # 各升级目标计数
    for tgt in "BCEA":
        n = sum(1 for r in rows if r["suspect"] == tgt)
        ns = sum(1 for r in rows if r["suspect"] == tgt and r["conf"] == "strong")
        if n:
            print(f"    →{tgt}: {n} 条 (strong {ns})")
    # strong 抽样
    print(f"\n===== strong 抽样(GT矛盾+双证同向,最硬) =====")
    for r in [x for x in rows if x["conf"] == "strong"][:a.sample]:
        print(f"[{r['gt_rt']}→{r['suspect']}] cue『{r['cue']}』 model={r['model_rt']} gem={r['gem_rt']} sk={r['gt_sk']}")
        print(f"   {r['desc'][:160]}")


if __name__ == "__main__":
    main()
