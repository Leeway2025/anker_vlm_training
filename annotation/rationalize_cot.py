"""CoT 重标注(给定标签的 rationalization 模式,带防硬编闸)。

由 label_euno_wds(客户已跑通的 WDS 盲判标注器)改造: 分片流式读取/
16 帧 image part/断点续跑/并发骨架全部复用,差异只有两点:
  1. prompt 换为"核验"立场 —— 把 GT 标签给 Gemini,让它判断标签是否
     说得通: 说得通(不要求唯一解释)→ 写推理链;只有证据明确矛盾、
     或需要编造画面外细节才 unsupported。偏向采信标签,但不许硬编。
  2. 产出直接是 asset_C 格式 + unsupported 错标嫌疑清单。

与盲判白名单(实测通过率 ~37%,偏简单样本)互补: 本脚本给全量样本
配推理链,覆盖率与难样本浓度双升。

产出:
  --out          原始记录(断点续跑锚点,含 evidence/verdict)
  --asset-out    asset_C 格式 {"video_id","reasoning_chain"}(仅 supported)
  <asset-out>.unsupported_ids.txt   错标嫌疑清单(可反哺训练集清洗)
  <out>.error.jsonl                 3 次重试仍失败的样本

失败重推理: 单条自动重试 3 次,仍失败写 <out>.error.jsonl;
**原命令原样再跑一遍即为二次重推理** —— 已成功(在 --out 里)的自动
跳过,循环到 error 不再减少为止。

用法:
  python -m annotation.rationalize_cot --wds-dir /data/shards \
      --labels DATA/labels.jsonl --out rat_pass.jsonl \
      --asset-out DATA_assets_rat/asset_C_reasoning.jsonl \
      --workers 50          # 客户配额 20~50;默认 50
  # --shards 0-3 可按分片切分多机并行;--limit 200 先试跑

吞吐参考: 单条 ~5-10s;50 并发 83k ≈ 3-4h。
判读: unsupported 率 <1% = 核验闸疑似未咬合(抽查 evidence);
      1%~20% = 正常;>30% = 标签质量问题或 prompt 过严,抽查再定。
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from annotation.label_euno_wds import iter_shard_samples, parse_shards  # noqa: E402
from data.euno_wds import read_json                                    # noqa: E402
from data.taxonomy import RT_NAMES                                      # noqa: E402

SK_NAMES = {
    "a": "Vehicle Access", "b": "Dog Walking", "c": "Kid Playing",
    "d": "Kid Studying", "e": "Leisure Activity", "f": "Home Chores",
    "g": "Visitor Arrival", "h": "Package Brought Home",
    "i": "Package Delivery", "j": "Person Falling", "k": "Leaving Porch",
    "l": "Approaching Porch", "m": "Other Normal Activity",
    "n": "Package Taken Away", "o": "Other Property Damage",
    "p": "Wildlife", "q": "Weapon Threat", "r": "Other Hazards",
    "s": "Loitering", "t": "Vehicle Anomaly", "u": "Unauthorized Entry",
}

FRAMES_PREFACE = ("The following 16 frames are uniformly sampled, in "
                  "temporal order, from one home-camera clip (frame 1 "
                  "earliest, frame 16 latest). Treat them as one video.\n")

# 核验立场(全英文,输出将作为 E2B 训练上下文): 标签不必是唯一解释,
# 是"说得通的解释之一"即 supported;明确矛盾/需编造细节才 unsupported。
PROMPT_TEMPLATE = """You are a home-security video analysis expert.

This clip was labeled by a human annotator:
  RoleType = {rt} ({rt_name})
  Sub-keyscene = {sk} ({sk_name})

Your job is to VERIFY whether this label is a reasonable reading of the clip, \
and if so, write the reasoning chain that justifies it.

Step 1 - Evidence: list 2-4 concrete observations actually visible in the frames \
(who/what object, action, location, motion direction). Describe only what is \
visible; never invent details.

Step 2 - Verdict:
- The label does NOT need to be the only possible interpretation. If the evidence \
is consistent with the label, or the label is a plausible interpretation among \
others, output verdict="supported" and write the reasoning chain.
- Output verdict="unsupported" ONLY if the evidence clearly contradicts the label, \
or supporting it would require inventing details that are not visible. Never fabricate.

Output STRICT JSON only, no other text:
{{
  "evidence": ["2-4 items, one short English sentence each, specific to object/action/location"],
  "verdict": "supported or unsupported",
  "reasoning_chain": "Only when supported, else empty string. About 30-40 English \
words: [Identity cues] clothing/familiarity/purposefulness [Scene cues] \
location/items/motion direction [Conclusion] one sentence linking the evidence to \
RoleType {rt} and Sub-keyscene {sk}. Cite only observations already listed in evidence."
}}

Label guide (same convention as the human annotation spec):
1. h vs n: same visible action (carrying a package away); decide by identity - \
enters/exits the home confidently, goes straight for the item, no scouting -> \
resident (h); otherwise n.
2. k vs l: judge the view first; in doorbell view, walking toward the camera then \
disappearing at the door = l, walking straight away outward = k.
3. RoleType C (Suspicious): behavior tied to s/t/u/n/q; prolonged aimless presence leans C.
4. RoleType E (Non-Human): no person visible; vehicle activity, pets/wildlife, fire are all E.
"""


def build_prompt(rt, sk):
    return PROMPT_TEMPLATE.format(
        rt=rt, sk=sk,
        rt_name=RT_NAMES.get(rt, "?"), sk_name=SK_NAMES.get(sk, "?"))


def validate_record(d):
    errs = []
    v = d.get("verdict")
    if v not in ("supported", "unsupported"):
        errs.append(f"bad verdict {v!r}")
    ev = d.get("evidence")
    if not isinstance(ev, list) or not ev:
        errs.append("evidence missing/empty")
    chain = d.get("reasoning_chain", "")
    if v == "supported":
        w = len(chain.split())
        if not (15 <= w <= 60):
            errs.append(f"chain {w} words (expect 30~40)")
        cjk = sum(1 for ch in chain if "一" <= ch <= "鿿")
        if chain and cjk > max(4, len(chain) * 0.1):
            errs.append("reasoning_chain must be English")
    return errs


def rationalize_frames_one(client, model_id, frames, prompt, temperature,
                           max_retries=3):
    """16 帧 JPEG bytes + 核验 prompt → Gemini。返回 (record, errs)。
    结构与 label_euno_wds.label_frames_one 一致,仅 prompt/校验不同。"""
    from google.genai import types
    contents = [FRAMES_PREFACE]
    contents += [types.Part.from_bytes(data=b, mime_type="image/jpeg")
                 for b in frames]
    contents.append(prompt)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_id, contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=8192,      # thinking 预算坑,勿降
                    response_mime_type="application/json"))
            d = json.loads(resp.text)
            errs = validate_record(d)
            if not errs:
                return d, None
            if attempt == max_retries - 1:
                return d, errs
        except Exception as e:
            if attempt == max_retries - 1:
                return None, [f"{type(e).__name__}: {e}"]
            time.sleep(5 * (attempt + 1))
    return None, ["unreachable"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds-dir", required=True, help="gs://… 或本地目录")
    ap.add_argument("--labels", required=True,
                    help="GT jsonl(含 labels 字段;同时限定标注范围)")
    ap.add_argument("--out", required=True, help="原始记录(断点续跑锚点)")
    ap.add_argument("--asset-out", required=True,
                    help="asset_C 格式产物(仅 supported)")
    ap.add_argument("--model", default="gemini-3.1-pro")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--shards", default=None, help="如 0-3 或 0,5,7")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=50,
                    help="并发标注数(按 API 配额调;客户配额 20~50)")
    ap.add_argument("--only-ids", default=None,
                    help="仅补采清单内的 video_id(如白名单外样本清单)")
    ap.add_argument("--vertex-project", default=None)
    ap.add_argument("--api-key", default=None,
                    help="Gemini API key(也可用环境变量 GOOGLE_API_KEY / "
                         "GEMINI_API_KEY)")
    ap.add_argument("--location", default="global")
    a = ap.parse_args()

    from google import genai
    client = genai.Client(vertexai=True, project=a.vertex_project,
                          location=a.location) if a.vertex_project \
        else genai.Client(api_key=a.api_key)   # None → SDK 读环境变量

    # GT 映射: video_id → (rt, sk);labels 复制行天然去重(dict 覆盖)
    gt = {}
    for line in open(a.labels, encoding="utf-8"):
        rec = json.loads(line)
        lab = rec.get("labels", rec)
        gt[rec["video_id"]] = (lab["role_type"], lab["sub_keyscene"])
    print(f"[scope] labels 覆盖 {len(gt)} keys")
    only = set(open(a.only_ids).read().split()) if a.only_ids else None

    done = set()
    if os.path.exists(a.out):      # 断点续跑/二次重推理: 成功的才算完成
        done = {json.loads(l)["video_id"] for l in open(a.out)}
        print(f"[resume] {len(done)} already done")
    os.makedirs(os.path.dirname(os.path.abspath(a.asset_out)), exist_ok=True)

    index = read_json(f"{a.wds_dir.rstrip('/')}/index.json")
    lock = threading.Lock()
    stats = {"ok": 0, "err": 0, "sup": 0, "unsup": 0}
    fout = open(a.out, "a", encoding="utf-8")
    ferr = open(a.out + ".error.jsonl", "a", encoding="utf-8")
    fasset = open(a.asset_out, "a", encoding="utf-8")
    funsup = open(a.asset_out + ".unsupported_ids.txt", "a",
                  encoding="utf-8")

    def work(item):
        key, frames = item
        rt, sk = gt[key]
        d, errs = rationalize_frames_one(client, a.model, frames,
                                         build_prompt(rt, sk),
                                         a.temperature)
        with lock:
            if d is not None and not errs:
                fout.write(json.dumps(
                    {"video_id": key, "gemini_model": a.model,
                     "temperature": a.temperature,
                     "gt": {"role_type": rt, "sub_keyscene": sk},
                     "rationalize_output": d}, ensure_ascii=False) + "\n")
                fout.flush()
                stats["ok"] += 1
                if d["verdict"] == "supported":
                    fasset.write(json.dumps(
                        {"video_id": key,
                         "reasoning_chain": d["reasoning_chain"]},
                        ensure_ascii=False) + "\n")
                    fasset.flush()
                    stats["sup"] += 1
                else:
                    funsup.write(key + "\n")
                    funsup.flush()
                    stats["unsup"] += 1
            else:
                ferr.write(json.dumps({"video_id": key, "errors": errs,
                                       "partial": d},
                                      ensure_ascii=False) + "\n")
                ferr.flush()
                stats["err"] += 1
            if (stats["ok"] + stats["err"]) % 50 == 0:
                r = stats["unsup"] / max(stats["ok"], 1)
                print(f"ok={stats['ok']} err={stats['err']} "
                      f"unsupported={stats['unsup']} ({r:.1%})", flush=True)

    def todo():
        n = 0
        for sid in parse_shards(a.shards, index):
            for key, frames in iter_shard_samples(a.wds_dir, sid):
                if key in done or key not in gt:
                    continue
                if only is not None and key not in only:
                    continue
                if a.limit and n >= a.limit:
                    return
                n += 1
                yield key, frames

    # 有界提交: 帧留在内存的任务数 ≤ 2×workers(pool.map 会吞光生成器)
    sem = threading.Semaphore(a.workers * 2)

    def bounded(item):
        try:
            work(item)
        finally:
            sem.release()

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for item in todo():
            sem.acquire()
            pool.submit(bounded, item)
    fout.close()
    ferr.close()
    fasset.close()
    funsup.close()

    r = stats["unsup"] / max(stats["ok"], 1)
    print(f"done: ok={stats['ok']} err={stats['err']} "
          f"supported={stats['sup']} unsupported={stats['unsup']} ({r:.1%})")
    if stats["ok"] and r < 0.01:
        print("⚠️ unsupported <1%: 核验闸疑似未咬合(模型在橡皮图章),"
              "抽查 evidence 是否真在描述画面")
    if stats["ok"] and r > 0.30:
        print("⚠️ unsupported >30%: 标签质量问题或 prompt 过严,抽查后再定")
    if stats["err"]:
        print(f"⚠️ {stats['err']} 条失败 → {a.out}.error.jsonl;"
              f"原命令再跑一遍即二次重推理(已成功样本自动跳过)")
    print(f"错标嫌疑清单 → {a.asset_out}.unsupported_ids.txt"
          f"(可反哺训练集清洗)")


if __name__ == "__main__":
    main()
