"""输出解析 + 校验(torch-free)。

职责(training_plan.md 13.3 / 14.2 节):
  1. 按"位置"解析 "{RT} | {SubKS} | {desc}"(不依赖大小写)
  2. 大小写自动矫正("c | C | ..." 这类错位防护)
  3. RT × SubKS 合法组合校验(部署侧兜底 + 评测健康指标)
  4. <think> 泄漏检测(Phase 5d 验收: 泄漏率必须 = 0%)
"""
from dataclasses import dataclass, asdict
from typing import Optional

RT_SET = set("ABCDE")
SK_SET = set("abcdefghijklmnopqrstu")

# 人物动作类 SubKS —— RT=E(无人)时逻辑上不可能
_HUMAN_ONLY_SK = set("bcdghjkl")
# RT=A(家人)不可能的 SubKS: n=偷包裹(定义即非住户)、u=未授权进入
_FAMILY_ILLEGAL_SK = set("nu")

# 安全关键 SubKS: 非法组合时建议升级为 C 告警(而非丢弃)
SAFETY_CRITICAL_SK = set("nqru")


def is_legal(rt: str, sk: str) -> bool:
    if rt == "E" and sk in _HUMAN_ONLY_SK:
        return False
    if rt == "A" and sk in _FAMILY_ILLEGAL_SK:
        return False
    return True


@dataclass
class ParseResult:
    ok: bool                      # 三字段齐全且分类字母有效
    rt: Optional[str] = None
    sk: Optional[str] = None
    desc: Optional[str] = None
    legal_combo: bool = True      # RT×SubKS 合法性
    case_fixed: bool = False      # 是否做过大小写矫正
    think_leak: bool = False      # 输出中出现 <think>
    raw: str = ""
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def parse_output(text: str) -> ParseResult:
    """按位置解析。字段顺序固定: 第 1 段=RT,第 2 段=SubKS,其余=desc。"""
    raw = text
    r = ParseResult(ok=False, raw=raw)

    if "<think>" in text or "</think>" in text:
        r.think_leak = True
        # 泄漏时仍尝试解析 think 之后的部分(容错,但泄漏本身计数)
        text = text.split("</think>")[-1]

    parts = [p.strip() for p in text.strip().split("|")]
    if len(parts) < 3:
        r.error = f"expected 3 pipe-separated fields, got {len(parts)}"
        return r

    rt_tok, sk_tok = parts[0], parts[1]
    desc = "|".join(parts[2:]).strip()   # desc 里万一含 "|" 全并回

    # 按位置取字段后做大小写矫正: RT 必大写,SK 必小写
    rt = rt_tok.upper() if len(rt_tok) == 1 else rt_tok
    sk = sk_tok.lower() if len(sk_tok) == 1 else sk_tok
    if rt != rt_tok or sk != sk_tok:
        r.case_fixed = True

    if rt not in RT_SET:
        r.error = f"invalid RT field: {rt_tok!r}"
        return r
    if sk not in SK_SET:
        r.error = f"invalid SubKS field: {sk_tok!r}"
        return r

    r.ok = True
    r.rt, r.sk, r.desc = rt, sk, desc
    r.legal_combo = is_legal(rt, sk)
    return r


def deployment_guard(res: ParseResult) -> dict:
    """端侧兜底建议(training_plan.md 14.2): 返回处置动作。"""
    if not res.ok:
        return {"action": "reject", "reason": res.error or "parse_failed"}
    if res.think_leak:
        return {"action": "flag_low_confidence", "reason": "think_leak"}
    if not res.legal_combo:
        if res.sk in SAFETY_CRITICAL_SK:
            return {"action": "escalate_to_C_alert",
                    "reason": f"illegal combo {res.rt}|{res.sk} on safety-critical scene"}
        return {"action": "flag_low_confidence",
                "reason": f"illegal combo {res.rt}|{res.sk}"}
    return {"action": "accept", "reason": ""}
