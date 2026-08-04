#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT 描述↔GT 标签 一致性 judge(纯文本, 不看视频)——"最方便的净收益"。
只读 GT 自己写的描述, 用 Gemini 判"描述是否【明确】违反 RoleType 规则"。
不用正则(解析不了 staff member/车钥匙 vs 家门钥匙语义), 不看帧(无 C 感知盲区,
因为我们信 GT 自己的描述, 不要 Gemini 去感知视频)。
口径(与 memory roletype-rules / rt-label-cleaning-low-roi 一致):
  只升级, 只在描述【肯定性】写明别的类别时才改; correct_rt 仅取 B/E/A/C, 不改 D。
  绝不降 C(描述没写威胁≠视频无威胁); 绝不因"缺家人线索"把 A 降 D。
用法(容器内, repo=/workspace):
  sudo docker exec tpu_train python /workspace/scripts/gt_desc_judge.py \
     --labels /data/labels_dedup.jsonl --out /data/gt_desc_judge.jsonl \
     --api-key 'AQ.Ab8RN6...' --model gemini-3.5-flash --workers 30 [--limit 200] [--stratified]
"""
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor

RULES = """RoleType rules (home-security doorbell/yard camera):
A = Family Member: a resident/household member. Positive cues: opens own door, uses a house key or access card, emerges from inside the home, returns home.
B = Staff: delivery/courier, uniform, technician/repair/plumber/electrician, gardener/landscaper, cleaner, firefighter/police officer, meter reader, or a worker with tools/ladder/company logo.
C = Suspicious Person: the description EXPLICITLY states a criminal-possibility ACTION — trying/prying/forcing a door or window, breaking in, climbing a fence/wall/gate, carrying a weapon, theft, or vandalism.
D = Unspecified: the fallback bucket when there is no positive Family/Staff/Suspicious evidence. NOT "stranger", NOT "delivery".
E = Non-Human: NO person present — vehicle-only activity (a car parked/reversing/driving by with nobody), an animal/pet/wildlife, or fire."""

INSTR = """You audit dataset labels. You are given ONLY the human-written DESCRIPTION of a clip and its assigned RoleType label. Decide whether the DESCRIPTION clearly CONTRADICTS the label. Judge strictly from the description text; do NOT imagine anything not written.

Flag "mislabel" ONLY when the description AFFIRMATIVELY describes something that clearly belongs to a DIFFERENT class:
- Uniform/delivery/courier/worker-with-tools described but label is A or D  -> correct_rt = B
- Vehicle-only / animal / no-person described but label is A/B/C/D          -> correct_rt = E
- Opens own door / house key / access card described but label is D          -> correct_rt = A
- Explicit criminal action (pry/force/break in/climb fence/weapon/theft) but label is A/B/D -> correct_rt = C

HARD CONSTRAINTS (only-upgrade):
- NEVER demote C. If the label is C and the description just doesn't mention a crime, output "ok" (the crime may be in the video but not written down).
- NEVER change a label to C unless the description explicitly states a criminal action.
- NEVER output correct_rt = D (D is only a fallback; do not move a label INTO D).
- If the description supports the label, or is merely vague/underspecified, output "ok".
- Correct_rt must be one of B, E, A, C only.

Output strict JSON: {"verdict":"ok" or "mislabel","correct_rt":"B|E|A|C or null","cue":"<=12-word quote from the description, or empty"}"""


def build_prompt(rt, desc):
    return f"{RULES}\n\n{INSTR}\n\nLABEL: {rt}\nDESCRIPTION: {desc}\n"


def judge_one(client, model_id, rt, desc, max_retries=3):
    from google.genai import types
    prompt = build_prompt(rt, desc)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_id, contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0, max_output_tokens=2048,
                    response_mime_type="application/json"))
            d = json.loads(resp.text)
            v = d.get("verdict"); cr = d.get("correct_rt")
            if v not in ("ok", "mislabel"):
                raise ValueError(f"bad verdict {v!r}")
            # 口径硬闸: 消灭模型越界(降C/入D/无中生有改C)
            if v == "mislabel":
                if cr not in ("A", "B", "C", "E"):
                    v, cr = "ok", None
                elif rt == "C":                 # 绝不降 C
                    v, cr = "ok", None
                elif cr == rt:                  # 没改动
                    v, cr = "ok", None
            return {"verdict": v, "correct_rt": cr if v == "mislabel" else None,
                    "cue": (d.get("cue") or "")[:120]}, None
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"{type(e).__name__}: {e}"
            time.sleep(4 * (attempt + 1))
    return None, "unreachable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/data/labels_dedup.jsonl")
    ap.add_argument("--out", default="/data/gt_desc_judge.jsonl")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stratified", action="store_true",
                    help="试跑: 每个 RT 均匀抽 limit/5 条(验精度用)")
    a = ap.parse_args()

    from google import genai
    # AQ. 前缀是 Vertex express key, 必须走 vertexai 通道(同 rationalize_cot/label_euno_wds)
    client = genai.Client(vertexai=True, api_key=a.api_key)

    recs = []
    for l in open(a.labels, encoding="utf-8"):
        d = json.loads(l); lab = d.get("labels") or {}
        recs.append((d["video_id"], lab.get("role_type"), lab.get("description") or ""))

    if a.limit and a.stratified:
        from collections import defaultdict
        per = max(1, a.limit // 5); by = defaultdict(list)
        for r in recs:
            by[r[1]].append(r)
        recs = [x for rt in "ABCDE" for x in by[rt][:per]]
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
        vid, rt, desc = r
        d, err = judge_one(client, a.model, rt, desc)
        with lock:
            if d is None:
                ferr.write(json.dumps({"video_id": vid, "err": err}) + "\n"); ferr.flush()
                stat["err"] += 1
            else:
                d.update({"video_id": vid, "gt_rt": rt, "desc": desc})
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
          f"| 错标率 {100*stat['mis']/max(1,tot):.1f}%", flush=True)


if __name__ == "__main__":
    main()
