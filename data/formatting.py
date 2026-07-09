"""目标序列构造与分类 token 加权(torch-free,可独立单元测试)。

核心职责:
  1. build_target: 按生产格式拼 "{RT} | {SubKS} | {desc}",返回字符级权重 span
  2. CoT 双模式: <think>…</think> 前缀(loss 权重 0)+ 答案
  3. char_spans_to_token_weights: 字符 span → token 级权重向量(基于 offset_mapping)

依据 training_plan.md 6.1/6.3/8 节:
  - 分类字母位置 loss 权重 = cls_token_weight(默认 4.0)
  - <think> 段权重 = 0(不反传)
  - description 权重 = 1.0
"""
from dataclasses import dataclass, field
from typing import List, Tuple

RT_LETTERS = set("ABCDE")
SK_LETTERS = set("abcdefghijklmnopqrstu")


@dataclass
class TargetSpec:
    text: str                                   # 完整目标字符串
    # (char_start, char_end, weight) —— 未覆盖区间默认权重 1.0
    weight_spans: List[Tuple[int, int, float]] = field(default_factory=list)


def build_target(rt: str, sk: str, desc: str, sep: str = "|",
                 cls_weight: float = 4.0) -> TargetSpec:
    """生产格式目标: "B | i | A delivery person ..."

    分类字母(及其后的分隔符)加权 cls_weight,description 权重 1.0。
    """
    if rt not in RT_LETTERS:
        raise ValueError(f"invalid RoleType letter: {rt!r}")
    if sk not in SK_LETTERS:
        raise ValueError(f"invalid Sub-keyscene letter: {sk!r}")
    desc = desc.strip()
    text = f"{rt}{sep}{sk}{sep}{desc}"
    # 加权区间: RT 字母 + 第一个分隔符 + SK 字母 + 第二个分隔符
    cls_end = len(rt) + len(sep) + len(sk) + len(sep)
    return TargetSpec(text=text, weight_spans=[(0, cls_end, cls_weight)])


def build_cot_target(rt: str, sk: str, desc: str, reasoning: str,
                     sep: str = "|", cls_weight: float = 4.0,
                     think_open: str = "<think>",
                     think_close: str = "</think>") -> TargetSpec:
    """Phase 5d 推理链模式目标:
        <think>{reasoning}</think>\n{RT} | {SubKS} | {desc}

    <think> 段(含标签)权重 0 —— forward 产生 hidden states 但不反传;
    答案段权重与 build_target 一致。
    reasoning 中若混入 think 标签字样(Gemini 输出不可控)会被消毒,
    防止嵌套标签破坏泄漏检测和答案定位。
    """
    answer = build_target(rt, sk, desc, sep, cls_weight)
    clean = reasoning.strip().replace(think_open, "").replace(think_close, "")
    prefix = f"{think_open}{clean}{think_close}\n"
    n = len(prefix)
    spans = [(0, n, 0.0)]
    for (s, e, w) in answer.weight_spans:
        spans.append((s + n, e + n, w))
    return TargetSpec(text=prefix + answer.text, weight_spans=spans)


def char_weight_at(spec: TargetSpec, pos: int) -> float:
    """查询单个字符位置的权重(span 未覆盖 → 1.0)。"""
    for (s, e, w) in spec.weight_spans:
        if s <= pos < e:
            return w
    return 1.0


def char_spans_to_token_weights(spec: TargetSpec,
                                offsets: List[Tuple[int, int]]) -> List[float]:
    """把字符级权重映射到 token 级。

    offsets: tokenizer(..., return_offsets_mapping=True) 给出的
             每个 token 的 (char_start, char_end),仅目标段的 token。
    规则: token 覆盖区间内的最大权重(保证分类字母 token 一定拿到高权重,
          即使 tokenizer 把 "B |" 合成一个 token)。
    (0,0) 的特殊 token → 权重 0。
    """
    out = []
    for (s, e) in offsets:
        if e <= s:            # special token
            out.append(0.0)
            continue
        w = max(char_weight_at(spec, p) for p in range(s, e))
        out.append(w)
    return out


def format_reasoning(identity_clues: str, scene_clues: str,
                     conclusion: str) -> str:
    """把 Gemini 资产 C 的三段推理拼成训练用文本(与 annotation_spec 3.2 对齐)。"""
    return (f"[身份线索] {identity_clues.strip()} "
            f"[场景线索] {scene_clues.strip()} "
            f"[结论] {conclusion.strip()}")
