"""类别体系单一来源(torch-free)。

权威依据: Anker场景数据定义0706.pdf + annotation_spec.md 3.2 节。
gemini_labeler(打标)与 training/common.AuxHeads(消费)都从这里取词表,
保证两端枚举一致。
"""

RT_LETTERS = "ABCDE"
SK_LETTERS = "abcdefghijklmnopqrstu"

RT_NAMES = {
    "A": "Family Member", "B": "Staff", "C": "Suspicious Person",
    "D": "Unspecified", "E": "Non-Human",
}

# SubKS → KeyScene 6 大类(父类头 + 聚合指标共用)
KS_GROUP = {**{c: "Normal" for c in "abcdefghijklm"},
            "n": "PropDmg", "o": "PropDmg",
            "p": "LifeThreat", "q": "LifeThreat", "r": "LifeThreat",
            "s": "Loiter", "t": "VehAnom", "u": "UnauthEntry"}
KS_CLASSES = sorted(set(KS_GROUP.values()))          # 6 类

SAFETY_SK = "qrunj"          # 安全关键 SubKS(监控 Recall)

# ---------- 辅助头属性词表(7 头;顺序即类别索引) ----------
AUX_VOCABS = {
    "clothing":  ["uniform", "casual", "reflective", "not_applicable"],
    "held_items": ["package", "tool", "weapon", "leash", "none", "other"],
    "posture":   ["standing", "walking", "bending", "sitting", "running",
                  "not_applicable"],
    "action":    ["walking", "lingering", "falling", "climbing",
                  "placing_item", "taking_item", "playing", "working",
                  "none"],
    "view_type": ["doorbell", "non_doorbell"],
    "sub_role_type": ["Kid", "Elder", "OtherFamily", "DeliveryPerson",
                      "Police", "ServiceWorker", "Passerby", "Unknown",
                      "NotApplicable"],
    "dwell":     ["passes_through", "brief_stop", "prolonged_stay"],
}
AUX_HEAD_ORDER = list(AUX_VOCABS.keys())


def aux_label_index(head: str, value: str) -> int:
    """属性值 → 类别索引;未知值返回 -100(loss 忽略)。"""
    try:
        return AUX_VOCABS[head].index(value)
    except (ValueError, KeyError):
        return -100


def view_type_from_resolution(width: int, height: int,
                              doorbell_res=((1600, 2200),)) -> str:
    """客户数据免费打标规则(training_plan 4.2 节,已确认):
    1600×2200 → doorbell;16:9 横屏 → non_doorbell;其他 → None(Gemini 兜底)
    """
    for (w, h) in doorbell_res:
        if (width, height) == (w, h):
            return "doorbell"
    if height and abs(width / height - 16 / 9) < 0.05:
        return "non_doorbell"
    return None
