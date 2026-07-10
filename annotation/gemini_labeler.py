"""Gemini 3.1 Pro 标注管线(框架无关;双轨: 我方代理集 / 客户 1M 数据)。

产出与 annotation_spec.md 3.2 对齐:
  attributes(7 组,词表来自 data/taxonomy.py,与训练侧单一来源)
  reasoning_chain(三段式)
  predictions(RT 字母 / SubKS 字母 / ~25 词英文 description)

用法:
  python -m annotation.gemini_labeler --labels labels.jsonl \
      --video-root videos/ --out pass1.jsonl --temperature 0.1
  # 双次一致性: 再以 --temperature 0.4 跑 pass2,过 consistency_filter
"""
import json
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.taxonomy import AUX_VOCABS, RT_LETTERS, SK_LETTERS  # noqa: E402

PROMPT_TEMPLATE = """你是家庭安防视频分析专家。输入是一段家用摄像头(门铃或院子/云台)视频。
严格输出 JSON,不要任何其他文本。

{{
  "attributes": {{
    "clothing": "{clothing} 中选一",
    "held_items": "{held_items} 中选一",
    "posture": "{posture} 中选一",
    "action": "{action} 中选一",
    "view_type": "{view_type} 中选一(doorbell=门铃近距正对门廊视角)",
    "sub_role_type": "{sub_role_type} 中选一(无人场景选 NotApplicable)",
    "dwell": "{dwell} 中选一(passes_through=直接经过; brief_stop=停留<10s; prolonged_stay=长时间逗留或反复出现)"
  }},
  "reasoning_chain": "⚠️ 必须用英文书写(约 30~40 个英文单词;该文本将作为小模型的推理上下文,小模型英文能力更强): [Identity cues] clothing/familiarity/purposefulness [Scene cues] location/items/motion direction [Conclusion] one sentence linking evidence to RoleType and Sub-keyscene",
  "predictions": {{
    "role_type": "A~E 单个大写字母 (A=Family, B=Staff, C=Suspicious, D=Unspecified, E=Non-Human)",
    "sub_keyscene": "a~u 单个小写字母 (a=VehicleAccess b=DogWalking c=KidPlaying d=KidStudying e=Leisure f=HomeChores g=VisitorArrival h=PackageBroughtHome i=PackageDelivery j=PersonFalling k=LeavingPorch l=ApproachingPorch m=OtherNormal n=PackageTakenAway o=OtherPropertyDamage p=Wildlife q=WeaponThreat r=OtherHazards s=Loitering t=VehicleAnomaly u=UnauthorizedEntry)",
    "description": "约 25 个英文单词的客观描述"
  }}
}}

特别注意:
1. h vs n 视觉动作相同(拿包裹离开),以身份区分: 开门熟练/从屋内出现/径直取件/无张望 → 住户(h);否则 n
2. k vs l 先判 view_type,门铃视角按"正面走来消失=l / 径直往外=k"
3. C 类: 行为与 s/t/u/n/q 关联时选 C;dwell=prolonged_stay 且无目的 → 倾向 C
4. E 类: 画面无人;车辆活动/宠物/火灾均属 E
5. description 客观陈述,不推测意图
"""


def build_prompt():
    return PROMPT_TEMPLATE.format(
        **{k: "/".join(v) for k, v in AUX_VOCABS.items()})


def validate_record(d):
    """格式校验(annotation_spec 4.2): 枚举合法性 + 字母合法性。"""
    errs = []
    p = d.get("predictions", {})
    if p.get("role_type") not in set(RT_LETTERS):
        errs.append(f"bad role_type {p.get('role_type')!r}")
    if p.get("sub_keyscene") not in set(SK_LETTERS):
        errs.append(f"bad sub_keyscene {p.get('sub_keyscene')!r}")
    desc = p.get("description", "")
    if not (10 <= len(desc.split()) <= 40):
        errs.append(f"desc length {len(desc.split())} words")
    chain = d.get("reasoning_chain", "")
    cjk = sum(1 for ch in chain if "\u4e00" <= ch <= "\u9fff")
    if chain and cjk > max(4, len(chain) * 0.1):
        errs.append("reasoning_chain 须英文(将作为 E2B 的 think 上下文,"
                    "小模型英文更强;中文占比过高)")
    attrs = d.get("attributes", {})
    for head, vocab in AUX_VOCABS.items():
        if attrs.get(head) not in vocab:
            errs.append(f"bad attr {head}={attrs.get(head)!r}")
    return errs


def label_one(client, model_id, video_path, prompt, temperature,
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
                    # thinking 系模型(2.5-pro/3.1-pro)的思考 token 也消耗
                    # 此预算,900 会导致 JSON 截断(真 API 集成测试发现)
                    max_output_tokens=8192,
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
    ap.add_argument("--labels", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-3.1-pro")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vertex-project", default=None,
                    help="走 Vertex 时的 GCP 项目(不传则用 GOOGLE_API_KEY)")
    ap.add_argument("--location", default="global")
    a = ap.parse_args()

    from google import genai
    if a.vertex_project:
        client = genai.Client(vertexai=True, project=a.vertex_project,
                              location=a.location)
    else:
        client = genai.Client()      # GOOGLE_API_KEY / ADC
    prompt = build_prompt()

    done = set()
    if os.path.exists(a.out):        # 断点续跑
        for line in open(a.out, encoding="utf-8"):
            done.add(json.loads(line)["video_id"])
    print(f"resume: {len(done)} already labeled")

    n_ok = n_err = 0
    with open(a.out, "a", encoding="utf-8") as fout, \
         open(a.out + ".errors", "a", encoding="utf-8") as ferr:
        for line in open(a.labels, encoding="utf-8"):
            rec = json.loads(line)
            vid = rec["video_id"]
            if vid in done:
                continue
            if a.limit and n_ok >= a.limit:
                break
            path = os.path.join(a.video_root,
                                os.path.basename(rec.get("video_uri", vid)))
            d, errs = label_one(client, a.model, path, prompt, a.temperature)
            if d is not None and not errs:
                fout.write(json.dumps(
                    {"video_id": vid, "gemini_model": a.model,
                     "temperature": a.temperature, "gemini_output": d},
                    ensure_ascii=False) + "\n")
                fout.flush()
                n_ok += 1
            else:
                ferr.write(json.dumps(
                    {"video_id": vid, "errors": errs, "partial": d},
                    ensure_ascii=False) + "\n")
                ferr.flush()
                n_err += 1
            if (n_ok + n_err) % 100 == 0:
                print(f"ok={n_ok} err={n_err}")
    print(f"done: ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
