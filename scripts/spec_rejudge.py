#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""规范版重审(judge v3, 08-04): 按客户标注规范原文 /home/nas-tpu-poc/label.md
重新裁决 v1 judge 的全部 flag。只做 keep/retract 二choice, 不新增方向。
规范关键条款(与常识相反, 必须硬编码):
  - B Staff 硬条件=制服/邮包/快递车/完整投递流程; 工具/割草/干活≠Staff
    ("开放院子做家务可以给家人, 工作人员除外")
  - 热区: 远处(across the street/distance/neighbor)的偷盗/异常/制服工
    一律=其他正常+路人 → 基于远处事件的改判全部撤回
  - 非法入侵要"长时间推拽"; 拧一次门把手跑走不算; 轻易开门进去不算
  - 偷包裹必须看见拿起带走动作
  - p 野生动物=熊/鳄鱼/蛇/野猪/狼(危险动物); 鹿/浣熊/鸟/松鼠=普通动物
  - b 遛狗=家人牵狗出入院子; 路人遛狗=休闲
  - i 投递必须看见完整放下动作, 否则其他正常
  - 上车关门开走=远离门廊(非上下车); 下车走向门=靠近门廊
用法: python scripts/spec_rejudge.py --dim rt --split train --api-key AQ...
"""
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor

SPEC_RT = """Customer annotation SPEC (authoritative, overrides common sense):
- Family Member (A): opens own door / house key / emerges from inside / returns home. GENEROUS rules: person walking frontally toward doorbell camera and disappearing = family entering; normal activity inside an enclosed yard = family; YARD WORK IN AN OPEN YARD CAN BE FAMILY (unless uniformed staff).
- Staff (B) — courier rule (UPDATED by customer): a GENUINE delivery action qualifies as courier REGARDLESS of clothing — many small North-American couriers wear plain clothes. Genuine delivery = arrives with a package/mail/food and places it at the door/porch or hands it over (may photograph). Uniform / mail bag / marked delivery vehicle also still qualify. Take-away direction (unchanged old rule, uniform STILL matters here): a person WITH courier evidence (uniform / mail bag / delivery vehicle / delivered their own package first) who takes a package away = still Staff (B, scene=package-taken); a person with NO courier evidence who ONLY takes a package away = theft -> Suspicious (C). BUT: carrying tools / ladder / mowing / cleaning / yard work alone still does NOT make Staff (yard work in an open yard can be family); and merely walking around holding a package without a visible placement is NOT delivery.
- Suspicious (C) = explicitly OCCURRING risk only: sustained forced-entry attempts (a single knock or one handle-jiggle then leaving is NOT enough; easily opening an unlocked gate and walking in is NOT intrusion), package theft (must SEE the taking-away), weapon / physical fight, vehicle break-in (night/masked/furtive; a natural daytime interaction with a car is NOT), vandalism.
- HOT ZONE rule: any event happening FAR from the property (across the street, down the road, in the distance, at a neighbor's) = Other Normal + Passerby (label D) REGARDLESS of content — even theft, vehicle anomaly, or uniformed workers far away.
- Non-Human (E): truly no person present."""

SPEC_SK = """Customer annotation SPEC (authoritative, overrides common sense):
- Package Delivery (i): must SEE the complete putting-down / handing-over action; carrying a package around without visible placement = Other Normal (m).
- Vehicle Access (a): person actively interacting with car (door/trunk/loading). BUT: if person gets in, closes the door and DRIVES AWAY = Leaving Porch (k), NOT (a); if person exits car and walks toward the door/porch = Approaching Porch (l), NOT (a). In those two cases the original label may be correct.
- Dog Walking (b): FAMILY member walking dog out of / into the yard. A passerby walking a dog = Leisure (e).
- Wildlife (p): DANGEROUS animals only — bear, crocodile/alligator, snake, wild boar, wolf. Deer, raccoon, squirrel, birds, cats, dogs = ordinary animals (m for non-human scenes).
- Package Pickup (h): resident takes package toward door / inside.
- Kid Playing (c): real play (toys, sports, playing with pet); merely standing/walking/suddenly running is NOT playing.
- HOT ZONE rule: events far from the property = Other Normal regardless of content."""

INSTR = """You re-audit previously proposed label corrections against the SPEC above. You are given: the original label, the proposed correction, and the human-written DESCRIPTION. Decide:
- "keep"    -> the correction is justified UNDER THE SPEC (description affirmatively shows spec-defined evidence for the corrected class, and no spec rule protects the original label)
- "retract" -> the correction violates the spec (e.g. Staff without uniform evidence, far-away/hot-zone event, single door-knock called intrusion, ordinary animal called wildlife, drives-away case called vehicle access, passerby dog walk called dog walking), OR the description is too vague to be sure.
When in doubt, retract (precision over recall).
Output strict JSON: {"decision":"keep" or "retract","reason":"<=15 words citing the spec rule"}"""


def build_prompt(spec, orig, corr, desc):
    return (f"{spec}\n\n{INSTR}\n\nORIGINAL LABEL: {orig}\n"
            f"PROPOSED CORRECTION: {corr}\nDESCRIPTION: {desc}\n")


def rejudge_one(client, model_id, spec, orig, corr, desc, max_retries=3):
    from google.genai import types
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=[build_prompt(spec, orig, corr, desc)],
                config=types.GenerateContentConfig(
                    temperature=0.0, max_output_tokens=1024,
                    response_mime_type="application/json"))
            d = json.loads(resp.text)
            if d.get("decision") not in ("keep", "retract"):
                raise ValueError("bad decision")
            return d, None
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                return None, f"{type(e).__name__}: {e}"
            time.sleep(4 * (attempt + 1))
    return None, "unreachable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", required=True, choices=["rt", "sk"])
    ap.add_argument("--flags", required=True,
                    help="v1 judge 输出(只重审其中 verdict=mislabel 的)")
    ap.add_argument("--keep-ids", default="",
                    help="可选: 只重审此清单内的 video_id(如现行剔除清单)")
    ap.add_argument("--corr-filter", default="",
                    help="可选: 只重审改判目标为该值的 flag(如 B —— 快递新规补审)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()

    spec = SPEC_RT if a.dim == "rt" else SPEC_SK
    kf = "correct_rt" if a.dim == "rt" else "correct_sk"
    of = "gt_rt" if a.dim == "rt" else "gt_sk"

    scope = None
    if a.keep_ids and os.path.exists(a.keep_ids):
        scope = {x.strip() for x in open(a.keep_ids) if x.strip()}
    todo = []
    for l in open(a.flags, encoding="utf-8"):
        j = json.loads(l)
        if j.get("verdict") != "mislabel":
            continue
        if scope is not None and j["video_id"] not in scope:
            continue
        if a.corr_filter and j[kf] != a.corr_filter:
            continue
        todo.append(j)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["video_id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [j for j in todo if j["video_id"] not in done]
    print(f"[scope] 待重审 {len(todo)}(已完成 {len(done)})", flush=True)

    from google import genai
    client = genai.Client(vertexai=True, api_key=a.api_key)
    lock = threading.Lock()
    stat = {"keep": 0, "retract": 0, "err": 0}
    fout = open(a.out, "a", encoding="utf-8")

    def work(j):
        d, err = rejudge_one(client, a.model, spec, j[of], j[kf],
                             j.get("desc") or "")
        with lock:
            if d is None:
                stat["err"] += 1
            else:
                fout.write(json.dumps({
                    "video_id": j["video_id"], "orig": j[of],
                    "corr": j[kf], "decision": d["decision"],
                    "reason": d.get("reason", "")[:120]},
                    ensure_ascii=False) + "\n")
                fout.flush()
                stat[d["decision"]] += 1
            n = sum(stat.values())
            if n % 200 == 0:
                print(f"  {n}/{len(todo)} keep={stat['keep']} "
                      f"retract={stat['retract']} err={stat['err']}",
                      flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    fout.close()
    print(f"[done] keep={stat['keep']} retract={stat['retract']} "
          f"err={stat['err']}", flush=True)


if __name__ == "__main__":
    main()
