#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT 描述↔SubKS 标签 一致性 judge(纯文本, 不看视频)—— RT 版(gt_desc_judge.py)
的 SubKS 姊妹篇。口径(08-04 与用户逐条对齐, 6 决策点):
  ① m=兜底: 绝不改进 m; m 改出去仅限物证类。
  ② k/l(离开/靠近门廊)不判: 文本方向不可靠, k/l/m 互不改判。
  ③ 非 Normal 父类(n/o/q/r/s/t/u)+ j 绝不降级(描述中性≠视频无事件);
     升进这些类仅当描述明确写了(且目标仅开放 n/j 两个物证类)。
  ④ RT 一起喂, 只用 legal_combo_matrix 的 ❌ 格: C|h(可疑人员≠拿包裹回家)→ n;
     ⚠️ 格一律放过。n 目标遇 gt_rt=A 拒绝(住户拿包裹=h 正确)。
  ⑤ 只判物证锚类: a车辆/b遛狗/c玩耍/d学习/h/i/n包裹/j摔倒/p野生动物;
     e↔f↔g↔m 软边界一律不判。宁漏抓, 抓到全是铁证。
  ⑥ 输出带 correct_sk, test 诊断与未来清训练集两用。
用法(容器内):
  sudo docker exec tpu_train python /workspace/scripts/gt_desc_judge_sk.py \
     --labels /data/labels_test.jsonl --out /data/gt_desc_judge_sk_test.jsonl \
     --api-key 'AQ...' [--limit 420 --stratified]
"""
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor

# 允许的改判目标(物证类, ⑤): 绝不含 m/k/l(①②), 不含 e/f/g 软类,
# 非 Normal 侧只开放 n(需⑤铁证+④身份闸)与 j(描述明确摔倒)
ALLOWED_TARGETS = set("abcdhijnp")
# 绝不降级的源(③): 非 Normal 父类 + j
NEVER_DEMOTE = set("jnoqrstu")

RULES = """Sub-Keyscene (SK) classes for home-security doorbell/yard cameras:
a = Vehicle Visit: a PERSON getting in/out of a car, arriving/leaving by car.
b = Dog Walking: a person walking a dog (leash).
c = Child Playing.
d = Child Studying.
e = Leisure: adult relaxing (sitting, chatting, phone, smoking...).
f = Housework: chores/yard work (gardening, cleaning, watering, carrying...).
g = Visitor Arrival: a visitor arrives (knocks, rings doorbell, waits at door).
h = Package Brought Home: a RESIDENT retrieves a package and brings it inside.
i = Package Delivery: a courier/delivery worker DROPS OFF a package.
j = Person Falling: a person falls to the ground.
k = Leaving Porch / l = Approaching Porch: direction-based walking classes.
m = Other Normal Activity: the fallback bucket for generic/unclear activity.
n = Package Taken Away: a NON-resident takes a package away (theft-like).
o = Property Damage. p = Wildlife (animal, no person). q = Weapon Threat.
r = Other Danger/Disaster. s = Loitering. t = Vehicle Anomaly.
u = Unauthorized Entry.
RoleType context: A=family/resident, B=staff/courier, C=suspicious person,
D=unknown person, E=non-human."""

INSTR = """You audit dataset labels. You are given ONLY the human-written DESCRIPTION of a clip plus its RoleType (RT) and Sub-Keyscene (SK) labels. Decide whether the DESCRIPTION clearly CONTRADICTS the SK label. Judge strictly from the description text; do NOT imagine anything not written.

Flag "mislabel" ONLY when the description AFFIRMATIVELY describes hard physical evidence of a DIFFERENT class:
- dog on a leash / walking a dog described, but SK is not b               -> correct_sk = b
- child playing described, but SK is not c                                 -> correct_sk = c
- child studying/doing homework described, but SK is not d                 -> correct_sk = d
- courier/delivery worker dropping off a package, but SK is e/f/g/h/m      -> correct_sk = i
- resident (RT=A) retrieving own package / bringing it inside, SK is e/f/g/i/m -> correct_sk = h
- someone EXPLICITLY takes a package AWAY from the property (and RT is not A), but SK is a Normal class (a-m) -> correct_sk = n
- RT=C (suspicious person) with SK=h is a contradiction (a suspicious person does not "bring a package home") -> correct_sk = n
- a person EXPLICITLY falls to the ground, but SK is not j                 -> correct_sk = j
- person getting in/out of a car / arriving-leaving by car, but SK is e/f/m -> correct_sk = a
- ONLY an animal / wildlife present (no person), but SK is a Normal a-m class -> correct_sk = p

HARD CONSTRAINTS (precision over recall):
- The evidence must be the MAIN/CENTRAL activity of the clip. If the description contains multiple subjects or activities and the labeled class plausibly matches ANY of them, output "ok" (the label may track the primary subject).
- Do NOT flag correct_sk = a when getting in/out of the car merely starts or ends another activity (loading tools, unloading bags, doing chores then driving away) — the other activity's label stands.
- NEVER output correct_sk = m, k, or l. m is a fallback (moving INTO it is not a correction); k/l depend on camera direction which text cannot verify.
- If SK is k, l, or m and the evidence is merely "walking/approaching/leaving", output "ok" (k/l/m boundaries are not judged).
- If SK is one of n, o, q, r, s, t, u, or j: ALWAYS output "ok" (the event may be in the video even if the description does not mention it — never demote).
- correct_sk = n requires the description to EXPLICITLY state the package is taken away/off the property, or the RT=C & SK=h contradiction; never when RT=A.
- correct_sk = j requires an explicit fall.
- e/f/g/m soft boundaries among themselves are NEVER judged (watering plants could be leisure or housework — subjective).
- If the description supports the label, or is vague/underspecified, output "ok".
- correct_sk must be one of: a b c d h i n j p.

Output strict JSON: {"verdict":"ok" or "mislabel","correct_sk":"<letter> or null","cue":"<=12-word quote from the description, or empty"}"""


def build_prompt(rt, sk, desc):
    return f"{RULES}\n\n{INSTR}\n\nRT LABEL: {rt}\nSK LABEL: {sk}\nDESCRIPTION: {desc}\n"


def judge_one(client, model_id, rt, sk, desc, max_retries=3):
    from google.genai import types
    prompt = build_prompt(rt, sk, desc)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_id, contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0, max_output_tokens=2048,
                    response_mime_type="application/json"))
            d = json.loads(resp.text)
            v = d.get("verdict"); cs = d.get("correct_sk")
            if v not in ("ok", "mislabel"):
                raise ValueError(f"bad verdict {v!r}")
            # 口径硬闸(镜像 prompt 的 HARD CONSTRAINTS, 消灭模型越界)
            if v == "mislabel":
                if cs not in ALLOWED_TARGETS:      # ①②⑤: 目标必须物证类
                    v, cs = "ok", None
                elif sk in NEVER_DEMOTE:           # ③: 绝不降级
                    v, cs = "ok", None
                elif cs == sk:                     # 没改动
                    v, cs = "ok", None
                elif cs == "n" and rt == "A":      # ④: 住户拿包裹=h, 非 n
                    v, cs = "ok", None
            return {"verdict": v, "correct_sk": cs if v == "mislabel" else None,
                    "cue": (d.get("cue") or "")[:120]}, None
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"{type(e).__name__}: {e}"
            time.sleep(4 * (attempt + 1))
    return None, "unreachable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/data/labels_test.jsonl")
    ap.add_argument("--out", default="/data/gt_desc_judge_sk_test.jsonl")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stratified", action="store_true",
                    help="试跑: 每个 SubKS 类均匀抽 limit/21 条(验精度用)")
    a = ap.parse_args()

    from google import genai
    # AQ. 前缀是 Vertex express key, 必须走 vertexai 通道
    client = genai.Client(vertexai=True, api_key=a.api_key)

    recs = []
    for l in open(a.labels, encoding="utf-8"):
        d = json.loads(l); lab = d.get("labels") or {}
        recs.append((d["video_id"], lab.get("role_type"),
                     lab.get("sub_keyscene"), lab.get("description") or ""))

    if a.limit and a.stratified:
        from collections import defaultdict
        per = max(1, a.limit // 21); by = defaultdict(list)
        for r in recs:
            by[r[2]].append(r)
        recs = [x for sk in "abcdefghijklmnopqrstu" for x in by[sk][:per]]
    elif a.limit:
        recs = recs[:a.limit]

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try: done.add(json.loads(l)["video_id"])
            except: pass
    todo = [r for r in recs if r[0] not in done]
    print(f"[scope] {len(recs)} 目标, {len(done)} 已完成, {len(todo)} 待判", flush=True)

    lock = threading.Lock()
    stat = {"ok": 0, "mis": 0, "err": 0}
    fout = open(a.out, "a", encoding="utf-8")
    ferr = open(a.out + ".error", "a", encoding="utf-8")

    def work(r):
        vid, rt, sk, desc = r
        d, err = judge_one(client, a.model, rt, sk, desc)
        with lock:
            if d is None:
                ferr.write(json.dumps({"video_id": vid, "err": err}) + "\n"); ferr.flush()
                stat["err"] += 1
            else:
                d.update({"video_id": vid, "gt_rt": rt, "gt_sk": sk, "desc": desc})
                fout.write(json.dumps(d, ensure_ascii=False) + "\n"); fout.flush()
                stat["mis" if d["verdict"] == "mislabel" else "ok"] += 1
            n = stat["ok"] + stat["mis"] + stat["err"]
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} ok={stat['ok']} mislabel={stat['mis']} err={stat['err']}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    fout.close(); ferr.close()
    tot = stat["ok"] + stat["mis"]
    print(f"[done] ok={stat['ok']} mislabel={stat['mis']} err={stat['err']} "
          f"| 嫌疑率 {100*stat['mis']/max(1,tot):.1f}%", flush=True)


if __name__ == "__main__":
    main()
