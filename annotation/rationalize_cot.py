"""CoT 重标注(给定标签的 rationalization 模式,带防硬编闸)。

与 gemini_labeler(盲判)互补: 把 GT 标签和视频一起给 Gemini,让它
为"标准答案"写推理链 —— 解决盲判白名单覆盖率低(实测 ~37%)且偏
简单样本的问题。防"看着答案硬编"的三道闸:
  1. prompt 强制先独立列画面证据,再对照标签(顺序不可颠倒);
  2. 证据不足/矛盾 → verdict=unsupported,不生成 reasoning_chain;
  3. unsupported 清单单独落盘 = 错标嫌疑名单(可反哺训练集清洗)。

产出:
  --out          原始记录(断点续跑锚点,含 evidence/verdict)
  --asset-out    asset_C 格式 {"video_id","reasoning_chain"}(仅 supported)
  <asset-out>.unsupported_ids.txt   错标嫌疑 video_id 清单

用法:
  python -m annotation.rationalize_cot --labels DATA/labels.jsonl \
      --video-root videos/ --out rat_pass.jsonl \
      --asset-out DATA_assets_rat/asset_C_reasoning.jsonl

判读: unsupported 率 <1% = 闸没咬合(prompt 被无视,重查);
      1%~25% = 正常;>30% = prompt 过严或标签质量问题,抽查再定。
"""
import json
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.taxonomy import RT_NAMES  # noqa: E402

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

PROMPT_TEMPLATE = """你是家庭安防视频分析专家。输入是一段家用摄像头(门铃或院子/云台)视频,
以及该视频的人工标注候选标签。你的任务分两步,顺序不可颠倒:

第一步(独立观察,此时忽略候选标签): 客观列出画面中的具体证据 —— 谁/什么物体、
什么动作、什么位置、运动方向。只写画面里真实可见的内容,禁止推测或补全画面外信息。

第二步(对照): 将候选标签与你列出的证据对照。
- 证据支持该标签 → verdict=supported,并写推理链;
- 证据不足、矛盾、或需要编造画面中不存在的细节才能自圆其说 → verdict=unsupported,
  不写推理链。宁可 unsupported,不许硬编。

候选标签: RoleType={rt} ({rt_name}) / Sub-keyscene={sk} ({sk_name})

严格输出 JSON,不要任何其他文本:
{{
  "evidence": ["2~4 条画面证据,每条一句英文,具体到物体/动作/位置"],
  "verdict": "supported 或 unsupported",
  "reasoning_chain": "仅 supported 时输出。⚠️ 英文,约 30~40 词: [Identity cues] clothing/familiarity/purposefulness [Scene cues] location/items/motion direction [Conclusion] one sentence linking the evidence to RoleType {rt} and Sub-keyscene {sk}. 只能引用 evidence 里已列出的观察。unsupported 时输出空字符串。"
}}

参考口径(与人工标注规范一致):
1. h vs n 视觉动作相同(拿包裹离开),以身份区分: 开门熟练/从屋内出现/径直取件/无张望 → 住户(h);否则 n
2. k vs l 先判视角,门铃视角按"正面走来消失=l / 径直往外=k"
3. C 类: 行为与 s/t/u/n/q 关联;长时间无目的逗留 → 倾向 C
4. E 类: 画面无人;车辆活动/宠物/火灾均属 E
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
        errs.append("evidence 缺失或为空")
    chain = d.get("reasoning_chain", "")
    if v == "supported":
        w = len(chain.split())
        if not (15 <= w <= 60):
            errs.append(f"chain {w} words(期望 30~40)")
        cjk = sum(1 for ch in chain if "一" <= ch <= "鿿")
        if chain and cjk > max(4, len(chain) * 0.1):
            errs.append("reasoning_chain 须英文")
    return errs


def rationalize_one(client, model_id, video_path, prompt, temperature,
                    max_retries=3):
    from google.genai import types
    for attempt in range(max_retries):
        try:
            video_bytes = open(video_path, "rb").read()
            resp = client.models.generate_content(
                model=model_id,
                contents=[
                    types.Part.from_bytes(data=video_bytes,
                                          mime_type="video/mp4"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=8192,   # thinking 系模型预算,勿调小
                    response_mime_type="application/json",
                ),
            )
            d = json.loads(resp.text)
            errs = validate_record(d)
            if not errs:
                return d, None
            if attempt == max_retries - 1:
                return d, errs
        except Exception as e:                        # 网络/解析/限流
            if attempt == max_retries - 1:
                return None, [f"{type(e).__name__}: {e}"]
            time.sleep(5 * (attempt + 1))
    return None, ["unreachable"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="GT jsonl(含 labels 字段)")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True, help="原始记录(断点续跑锚点)")
    ap.add_argument("--asset-out", required=True,
                    help="asset_C 格式产物(仅 supported)")
    ap.add_argument("--model", default="gemini-3.1-pro")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-ids", default=None,
                    help="仅补采清单内的 video_id(如白名单外样本清单)")
    ap.add_argument("--vertex-project", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--location", default="global")
    a = ap.parse_args()

    from google import genai
    if a.vertex_project:
        client = genai.Client(vertexai=True, project=a.vertex_project,
                              location=a.location)
    else:
        client = genai.Client(api_key=a.api_key)

    only = (set(open(a.only_ids).read().split())
            if a.only_ids else None)
    done = set()
    if os.path.exists(a.out):        # 断点续跑
        for line in open(a.out, encoding="utf-8"):
            done.add(json.loads(line)["video_id"])
    print(f"resume: {len(done)} already done")
    os.makedirs(os.path.dirname(os.path.abspath(a.asset_out)), exist_ok=True)
    unsup_path = a.asset_out + ".unsupported_ids.txt"

    seen = set()
    n_ok = n_err = n_sup = n_unsup = 0
    with open(a.out, "a", encoding="utf-8") as fout, \
         open(a.out + ".errors", "a", encoding="utf-8") as ferr, \
         open(a.asset_out, "a", encoding="utf-8") as fasset, \
         open(unsup_path, "a", encoding="utf-8") as funsup:
        for line in open(a.labels, encoding="utf-8"):
            rec = json.loads(line)
            vid = rec["video_id"]
            if vid in done or vid in seen:      # labels 可能含复制行,只标一次
                continue
            seen.add(vid)
            if only is not None and vid not in only:
                continue
            if a.limit and n_ok >= a.limit:
                break
            lab = rec.get("labels", rec)
            prompt = build_prompt(lab["role_type"], lab["sub_keyscene"])
            path = os.path.join(a.video_root,
                                os.path.basename(rec.get("video_uri", vid)))
            d, errs = rationalize_one(client, a.model, path, prompt,
                                      a.temperature)
            if d is not None and not errs:
                fout.write(json.dumps(
                    {"video_id": vid, "gemini_model": a.model,
                     "temperature": a.temperature,
                     "gt": {"role_type": lab["role_type"],
                            "sub_keyscene": lab["sub_keyscene"]},
                     "rationalize_output": d}, ensure_ascii=False) + "\n")
                fout.flush()
                n_ok += 1
                if d["verdict"] == "supported":
                    fasset.write(json.dumps(
                        {"video_id": vid,
                         "reasoning_chain": d["reasoning_chain"]},
                        ensure_ascii=False) + "\n")
                    fasset.flush()
                    n_sup += 1
                else:
                    funsup.write(vid + "\n")
                    funsup.flush()
                    n_unsup += 1
            else:
                ferr.write(json.dumps(
                    {"video_id": vid, "errors": errs, "partial": d},
                    ensure_ascii=False) + "\n")
                ferr.flush()
                n_err += 1
            if (n_ok + n_err) % 100 == 0:
                r = n_unsup / max(n_ok, 1)
                print(f"ok={n_ok} err={n_err} unsupported={n_unsup}"
                      f" ({r:.1%})")
    r = n_unsup / max(n_ok, 1)
    print(f"done: ok={n_ok} err={n_err} supported={n_sup} "
          f"unsupported={n_unsup} ({r:.1%})")
    if n_ok and r < 0.01:
        print("⚠️ unsupported <1%: 防硬编闸疑似未咬合(模型无视了对照要求),"
              "抽查 evidence 是否真在独立观察")
    if n_ok and r > 0.30:
        print("⚠️ unsupported >30%: prompt 过严或标签质量问题,抽查后再定")
    print(f"错标嫌疑清单 → {unsup_path}(可反哺训练集清洗)")


if __name__ == "__main__":
    main()
