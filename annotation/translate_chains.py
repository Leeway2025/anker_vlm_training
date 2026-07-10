# -*- coding: utf-8 -*-
"""
translate_chains.py —— pass1.jsonl 推理链中→英批量翻译(补救工具)

背景: 2026-07-10 前的标注 prompt 未强制 reasoning_chain 语言,Gemini 随
中文指令返回了中文推理链。推理链的**内容/逻辑本身有效**(Gemini 看视频
提炼的证据链),只是语言不对 —— 5d 阶段它是 Gemma E2B 的 think 上下文,
需要英文。本工具做**纯文本翻译**(不传任何视频帧),成本 ≈ 重标的 1~2%:
重标贵在 16 帧图像 token(~4k tok/条);翻译只有 ~150 tok/条文本。

用法(与标注器同一鉴权):
    python -m annotation.translate_chains \
        --in pass1.jsonl --out pass1_en.jsonl \
        --model gemini-2.5-flash-lite --vertex-project <P> [--batch 20]

行为:
- 已是英文的记录(CJK<10%)原样直通,不花钱;
- 中文链按 --batch 条/请求 打包翻译,JSON 数组进出,索引对齐;
- 输出记录 = 原记录 + reasoning_chain 替换为英文 + reasoning_chain_zh 留档;
- 断点续跑: --out 已有的 video_id 跳过;失败批写 .errors,重跑自动重试;
- 产出后直接喂 split_assets(无需改任何下游)。
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

SYS = """You are a precise translator for a video-surveillance ML pipeline.
Translate each Chinese reasoning chain into natural English.
FIDELITY IS THE TOP PRIORITY: translate EVERY piece of evidence — do not drop,
merge, add, or embellish any detail (counts like "multiple", qualifiers like
"most", negations like "no uniform or tools" must all survive). Be concise in
wording, but never at the cost of completeness.
If the source contains structure tags, convert them exactly:
[身份线索]→[Identity cues]  [场景线索]→[Scene cues]  [结论]→[Conclusion]
If the source has NO tags, do NOT add any.
Keep any trailing label like "→ D|m" byte-identical.
Input is a JSON array of strings; output MUST be a JSON array of the same
length, same order, translations only. No markdown, no commentary."""

JUDGE_SYS = """You are a strict translation auditor. For each item decide whether EN
is a faithful and complete translation of ZH: same evidence, same entities/actions/
qualifiers/negations, same conclusion, nothing added or dropped that changes meaning.
Structure-tag conversion ([身份线索]→[Identity cues] etc.) is expected, not an issue;
flag tags ADDED where ZH had none. Minor rewording is fine.
Output a JSON array, one object per input item, SAME order:
{"id": "<echo id>", "faithful": true/false, "issue": "<short reason if false>"}"""


TAG_MAP = {"[身份线索]": "[Identity cues]", "[场景线索]": "[Scene cues]",
           "[结论]": "[Conclusion]"}


def normalize_tags(t: str) -> str:
    for zh, en in TAG_MAP.items():
        t = t.replace(zh, en)
    return t


def cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if "一" <= ch <= "鿿") / len(s)


def get_chain(rec: dict) -> str:
    return (rec.get("gemini_output") or {}).get("reasoning_chain") \
        or rec.get("reasoning_chain") or ""


def set_chain(rec: dict, en: str, zh: str) -> None:
    tgt = rec.get("gemini_output") if isinstance(rec.get("gemini_output"), dict) else rec
    tgt["reasoning_chain"] = en
    tgt["reasoning_chain_zh"] = zh


def translate_batch(client, model: str, chains: list) -> list:
    need_tags = [("[身份线索]" in c or "[Identity cues]" in c) for c in chains]
    from google.genai import types
    rsp = client.models.generate_content(
        model=model,
        contents=json.dumps(chains, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=SYS, temperature=0.0,
            response_mime_type="application/json"),
    )
    out = json.loads(rsp.text)
    if not (isinstance(out, list) and len(out) == len(chains)):
        raise ValueError(f"batch shape mismatch: {len(chains)} in, "
                         f"{len(out) if isinstance(out, list) else type(out)} out")
    out = [normalize_tags(t) if isinstance(t, str) else t for t in out]
    for i, t in enumerate(out):
        if not isinstance(t, str) or cjk_ratio(t) > 0:
            raise ValueError(f"item {i} not English: {str(t)[:60]}")
        if need_tags[i] and ("[Identity cues]" not in t
                             or "[Conclusion]" not in t):
            raise ValueError(f"item {i} lost structure tags: {t[:60]}")
    usage = getattr(rsp, "usage_metadata", None)
    toks = ((usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)) \
        if usage else 0
    return out, toks


def judge_pairs(client, model: str, pairs: list) -> list:
    """pairs: [{id, zh, en}] → [{id, faithful, issue}](回显 id 校验防裁判错位)"""
    from google.genai import types
    verdicts = []
    for b0 in range(0, len(pairs), 10):
        batch = pairs[b0:b0 + 10]
        rsp = client.models.generate_content(
            model=model, contents=json.dumps(batch, ensure_ascii=False),
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYS, temperature=0.0,
                response_mime_type="application/json"))
        res = json.loads(rsp.text)
        if len(res) != len(batch):
            raise ValueError("judge batch shape mismatch")
        for want, got in zip(batch, res):
            if got.get("id") != want["id"]:
                raise ValueError(f"judge id echo mismatch: {got.get('id')}")
            verdicts.append(got)
    return verdicts


def retranslate_one(client, model: str, zh: str, issue: str) -> str:
    """不合格条目单条重译(带裁判意见,零错位风险)。"""
    from google.genai import types
    rsp = client.models.generate_content(
        model=model,
        contents=(f"Previous translation was rejected by an auditor: {issue}\n"
                  f"Translate again, complete and faithful:\n{zh}"),
        config=types.GenerateContentConfig(system_instruction=SYS,
                                           temperature=0.0))
    t = normalize_tags(rsp.text.strip().strip('"'))
    if cjk_ratio(t) > 0:
        raise ValueError(f"retranslation not English: {t[:60]}")
    return t


def verify_and_fix(client, model: str, out_path: str, rounds: int = 2) -> None:
    """翻译后自检: 裁判全量 → 不合格单条重译 → 复判;仍不合格写 .lowfidelity"""
    recs = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    def chain_obj(r):
        return r.get("gemini_output") if isinstance(r.get("gemini_output"), dict) else r
    idx = {r["video_id"]: r for r in recs}
    pending = [{"id": r["video_id"], "zh": chain_obj(r).get("reasoning_chain_zh", ""),
                "en": chain_obj(r)["reasoning_chain"]}
               for r in recs if chain_obj(r).get("reasoning_chain_zh")]
    low = []
    for rd in range(rounds):
        if not pending:
            break
        verdicts = judge_pairs(client, model, pending)
        bad = [p for p, v in zip(pending, verdicts) if not v["faithful"]]
        issues = {v["id"]: v.get("issue", "") for v in verdicts if not v["faithful"]}
        print(f"[verify] round {rd + 1}: {len(pending)} judged, "
              f"{len(bad)} unfaithful", file=sys.stderr)
        nxt = []
        for p in bad:
            if rd == rounds - 1:
                low.append({**p, "issue": issues[p["id"]]})
                continue
            try:
                en = retranslate_one(client, model, p["zh"], issues[p["id"]])
                chain_obj(idx[p["id"]])["reasoning_chain"] = en
                nxt.append({"id": p["id"], "zh": p["zh"], "en": en})
            except Exception as e:                    # noqa: BLE001
                low.append({**p, "issue": str(e)})
        pending = nxt
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(tmp).replace(out_path)
    if low:
        lp = Path(out_path + ".lowfidelity")
        with lp.open("w", encoding="utf-8") as fl:
            for x in low:
                fl.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[verify] {len(low)} 条复判仍不合格 → {lp}(建议人工抽看)",
              file=sys.stderr)
    else:
        print("[verify] 全部通过", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--vertex-project", default=None)
    ap.add_argument("--vertex-location", default="global")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过翻译后自动裁判(默认开启)")
    ap.add_argument("--verify-rounds", type=int, default=2)
    args = ap.parse_args()

    from google import genai
    client = (genai.Client(vertexai=True, project=args.vertex_project,
                           location=args.vertex_location)
              if args.vertex_project else genai.Client())

    done = set()
    outp = Path(args.out)
    if outp.exists():
        for ln in outp.open(encoding="utf-8"):
            try:
                done.add(json.loads(ln)["video_id"])
            except Exception:
                pass

    passthrough, todo = 0, []
    with outp.open("a", encoding="utf-8") as fo:
        for ln in open(args.inp, encoding="utf-8"):
            rec = json.loads(ln)
            if rec.get("video_id") in done:
                continue
            chain = get_chain(rec)
            if cjk_ratio(chain) <= 0.10:          # 已英文/空 → 直通,零成本
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                passthrough += 1
                continue
            todo.append(rec)

        total_tok, ok, failed = 0, 0, []
        for b0 in range(0, len(todo), args.batch):
            batch = todo[b0:b0 + args.batch]
            chains = [get_chain(r) for r in batch]
            for attempt in range(args.retries):
                try:
                    ens, toks = translate_batch(client, args.model, chains)
                    total_tok += toks
                    for r, en, zh in zip(batch, ens, chains):
                        set_chain(r, en, zh)
                        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fo.flush()
                    ok += len(batch)
                    break
                except Exception as e:               # noqa: BLE001
                    if attempt == args.retries - 1:
                        failed.extend(
                            {"video_id": r.get("video_id"), "error": str(e)}
                            for r in batch)
                    else:
                        time.sleep(2 * (attempt + 1))
            print(f"[translate] {ok}/{len(todo)} done, tokens={total_tok}",
                  file=sys.stderr)

    if failed:
        errp = Path(args.out + ".errors")
        with errp.open("w", encoding="utf-8") as fe:
            for f in failed:
                fe.write(json.dumps(f, ensure_ascii=False) + "\n")
        print(f"[translate] {len(failed)} failed → {errp}(重跑本命令自动重试)",
              file=sys.stderr)
    print(f"[translate] passthrough(已英文)={passthrough} translated={ok} "
          f"failed={len(failed)} total_tokens={total_tok}", file=sys.stderr)
    if not args.no_verify:
        verify_and_fix(client, args.model, args.out, rounds=args.verify_rounds)


if __name__ == "__main__":
    main()
