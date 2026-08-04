#!/usr/bin/env python3
"""
模型预测准确率计算工具 — Anker VLM 评测版。

适配格式:
  预测文件(jsonl): {"video_id": "...", "output": "C|n|desc"}
  GT 文件(jsonl):   {"video_id": "...", "labels": {"role_type": "C", "sub_keyscene": "n", ...}}

用法:
  python -m eval.metrics_anker \
      --pred outputs/phase5_sft_b/preds_test.jsonl \
      --gt   DATA/labels_test.jsonl \
      --out-dir eval_results/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

# ============================================================================
# 配置常量
# ============================================================================

# RoleType 代码 → 全名
ROLE_CODE_TO_NAME = {
    "A": "Family Member",
    "B": "Staff",
    "C": "Suspicious Person",
    "D": "Unspecified",
    "E": "Non-Human",
}
ROLE_NAME_TO_CODE = {v: k for k, v in ROLE_CODE_TO_NAME.items()}

# Sub-Keyscene 代码 → 全名
SK_CODE_TO_NAME = {
    "a": "Vehicle Access",       "b": "Dog Walking",
    "c": "Kid Playing",          "d": "Kid Studying",
    "e": "Leisure Activity",     "f": "Home Chores",
    "g": "Visitor Arrival",      "h": "Package Pickup",
    "i": "Package Delivery",     "j": "Person Falling",
    "k": "Leaving Porch",        "l": "Approaching Porch",
    "m": "Other Normal Activity","n": "Package Theft",
    "o": "Other Property Damage","p": "Wildlife",
    "q": "Weapon Threat",        "r": "Other Hazards",
    "s": "Loitering",            "t": "Vehicle Anomaly",
    "u": "Unauthorized Entry",
}
SK_NAME_TO_CODE = {v: k for k, v in SK_CODE_TO_NAME.items()}

# Sub-Keyscene → KeyScene 归属
SK_TO_KEY_MAPPING = {
    "a": "Normal Activity",    "b": "Normal Activity",
    "c": "Normal Activity",    "d": "Normal Activity",
    "e": "Normal Activity",    "f": "Normal Activity",
    "g": "Normal Activity",    "h": "Normal Activity",
    "i": "Normal Activity",    "j": "Normal Activity",
    "k": "Normal Activity",    "l": "Normal Activity",
    "m": "Normal Activity",
    "n": "Property Damage",    "o": "Property Damage",
    "p": "Life-Threatening Emergency",
    "q": "Life-Threatening Emergency",
    "r": "Life-Threatening Emergency",
    "s": "Loitering",
    "t": "Vehicle Anomaly",
    "u": "Unauthorized Entry",
}

# 展示顺序
ROLE_LABELS = [
    "Family Member", "Staff", "Suspicious Person",
    "Non-Human", "Unspecified",
]
KEYSCENE_LABELS = [
    "Normal Activity", "Property Damage", "Life-Threatening Emergency",
    "Loitering", "Vehicle Anomaly", "Unauthorized Entry",
]
SK_LABELS = [
    "Vehicle Access", "Dog Walking", "Kid Playing", "Kid Studying",
    "Leisure Activity", "Home Chores", "Visitor Arrival", "Package Pickup",
    "Package Delivery", "Person Falling", "Leaving Porch", "Approaching Porch",
    "Other Normal Activity", "Package Theft", "Other Property Damage",
    "Wildlife", "Weapon Threat", "Other Hazards",
    "Loitering", "Vehicle Anomaly", "Unauthorized Entry",
]

# 屏蔽类别 — 用全名(如 "Suspicious Person" / "Package Theft" / "Normal Activity")
IGNORED_CATEGORIES: Dict[str, List[str]] = {
    "role_type": [],
    "keyscene": [],
    "sub_keyscene": [],
}


# ============================================================================
# 输出解析
# ============================================================================

@dataclass
class ParseResult:
    """解析后的预测输出。"""
    ok: bool
    role: Optional[str] = None          # 代码 (A-E)
    sub_keyscene: Optional[str] = None   # 代码 (a-u)
    desc: Optional[str] = None
    error: Optional[str] = None

    @property
    def role_name(self) -> Optional[str]:
        return ROLE_CODE_TO_NAME.get(self.role) if self.role else None

    @property
    def sk_name(self) -> Optional[str]:
        return SK_CODE_TO_NAME.get(self.sub_keyscene) if self.sub_keyscene else None

    @property
    def keyscene_name(self) -> Optional[str]:
        return SK_TO_KEY_MAPPING.get(self.sub_keyscene) if self.sub_keyscene else None


def parse_output(text: str) -> ParseResult:
    """解析 "RT|SubKey|desc" 格式的预测字符串。"""
    if not text:
        return ParseResult(ok=False, error="empty output")

    # 处理  thinking 标签
    if " thinking" in text:
        # 剔除 think 段，取 response 之后的部分
        parts_after = text.split(" response")
        text = parts_after[-1] if len(parts_after) > 1 else text

    parts = [p.strip() for p in text.strip().split("|")]
    if len(parts) < 3:
        return ParseResult(ok=False, error=f"expected 3 fields, got {len(parts)}")

    rt_raw, sk_raw = parts[0], parts[1]
    desc = "|".join(parts[2:]).strip()

    rt = rt_raw.upper() if len(rt_raw) == 1 else rt_raw
    sk = sk_raw.lower() if len(sk_raw) == 1 else sk_raw

    if rt not in ROLE_CODE_TO_NAME:
        return ParseResult(ok=False, error=f"invalid RT: {rt_raw!r}")
    if sk not in SK_CODE_TO_NAME:
        return ParseResult(ok=False, error=f"invalid SubKS: {sk_raw!r}")

    return ParseResult(ok=True, role=rt, sub_keyscene=sk, desc=desc)


# ============================================================================
# 数据加载
# ============================================================================

@dataclass
class EvalRecord:
    """单条评测记录。"""
    video_id: str
    # GT (代码)
    gt_role: Optional[str] = None
    gt_sk: Optional[str] = None
    # 预测 (代码)
    pred_role: Optional[str] = None
    pred_sk: Optional[str] = None
    # 解析状态
    parse_ok: bool = True
    parse_error: Optional[str] = None


def load_jsonl(path: str) -> Dict[str, dict]:
    """加载 JSONL 文件，返回 {video_id: record}。"""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            vid = obj.get("video_id")
            if vid:
                data[vid] = obj
    return data


def load_records(pred_path: str, gt_path: str) -> List[EvalRecord]:
    """加载预测和 GT，按 video_id 对齐，返回评测记录列表。"""
    preds = load_jsonl(pred_path)
    gts = load_jsonl(gt_path)

    records = []
    n_not_found = 0

    for vid, gt_obj in gts.items():
        pred_obj = preds.get(vid)
        if pred_obj is None:
            # GT 中有的样本，预测中没有
            n_not_found += 1
            records.append(EvalRecord(
                video_id=vid,
                gt_role=gt_obj["labels"].get("role_type"),
                gt_sk=gt_obj["labels"].get("sub_keyscene"),
                parse_ok=False,
                parse_error="missing prediction",
            ))
            continue

        output = pred_obj.get("output", "")
        parsed = parse_output(output)

        records.append(EvalRecord(
            video_id=vid,
            gt_role=gt_obj["labels"].get("role_type"),
            gt_sk=gt_obj["labels"].get("sub_keyscene"),
            pred_role=parsed.role,
            pred_sk=parsed.sub_keyscene,
            parse_ok=parsed.ok,
            parse_error=parsed.error,
        ))

    if n_not_found:
        print(f"⚠️  预测中缺失的 GT 样本: {n_not_found}")

    return records


# ============================================================================
# 指标计算
# ============================================================================

@dataclass
class CategoryStats:
    """单类别统计。"""
    gt_count: int = 0
    pred_count: int = 0
    correct_count: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.gt_count if self.gt_count > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.correct_count / self.pred_count if self.pred_count > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.correct_count / self.gt_count if self.gt_count > 0 else 0.0


def compute_metrics(
    gt_list: List[str], pred_list: List[str]
) -> Tuple[float, Dict[str, CategoryStats]]:
    """计算总体准确率和每类别统计。"""
    stats: Dict[str, CategoryStats] = defaultdict(CategoryStats)

    total_correct = 0
    for gt, pred in zip(gt_list, pred_list):
        stats[gt].gt_count += 1
        stats[pred].pred_count += 1
        if gt == pred:
            stats[gt].correct_count += 1
            total_correct += 1

    overall_acc = total_correct / len(gt_list) if gt_list else 0.0
    return overall_acc, dict(stats)


# ============================================================================
# 报告输出
# ============================================================================

def build_table(
    stats: Dict[str, CategoryStats],
    labels: List[str],
    title: str,
) -> str:
    """构建类别统计表格。"""
    rows = []
    total_gt = total_pred = total_correct = 0

    for label in labels:
        s = stats.get(label)
        if s is None:
            continue
        rows.append([
            label, s.gt_count, s.pred_count, s.correct_count,
            f"{s.accuracy:.4f}", f"{s.precision:.4f}", f"{s.recall:.4f}",
        ])
        total_gt += s.gt_count
        total_pred += s.pred_count
        total_correct += s.correct_count

    # 还有不在 labels 中的类别（归入 Other）
    other_gt = other_pred = other_correct = 0
    for cat, s in stats.items():
        if cat not in labels:
            other_gt += s.gt_count
            other_pred += s.pred_count
            other_correct += s.correct_count
    if other_gt > 0:
        rows.append([
            "[Other]", other_gt, other_pred, other_correct,
            f"{other_correct / other_gt:.4f}" if other_gt else "N/A",
            f"{other_correct / other_pred:.4f}" if other_pred else "N/A",
            f"{other_correct / other_gt:.4f}" if other_gt else "N/A",
        ])
        total_gt += other_gt
        total_pred += other_pred
        total_correct += other_correct

    total_acc = total_correct / total_gt if total_gt else 0.0
    total_prec = total_correct / total_pred if total_pred else 0.0
    total_rec = total_correct / total_gt if total_gt else 0.0

    rows.append([
        "**Total**", total_gt, total_pred, total_correct,
        f"{total_acc:.4f}", f"{total_prec:.4f}", f"{total_rec:.4f}",
    ])

    headers = ["Category", "GT", "Pred", "Correct", "Acc", "Prec", "Recall"]
    return tabulate(rows, headers=headers, tablefmt="grid",
                    numalign="left", stralign="left")


def save_confusion_matrix(
    gt_list: List[str],
    pred_list: List[str],
    labels: List[str],
    save_path: str,
    title: str,
):
    """保存混淆矩阵 PNG。"""
    if not gt_list:
        return

    cm = confusion_matrix(gt_list, pred_list, labels=labels)

    figsize = max(10, len(labels) * 0.6)
    plt.figure(figsize=(figsize, figsize * 0.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  混淆矩阵 → {save_path}")


# ============================================================================
# 主流程
# ============================================================================

def run_evaluation(pred_path: str, gt_path: str, out_dir: str):
    """执行完整评测流程。"""
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. 加载数据 ──
    print("=" * 60)
    print("Anker VLM 预测准确率评测")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"预测: {pred_path}")
    print(f"GT:   {gt_path}")
    print()

    records = load_records(pred_path, gt_path)
    print(f"总样本数: {len(records)}")

    # 解析统计
    n_ok = sum(1 for r in records if r.parse_ok)
    n_fail = len(records) - n_ok
    print(f"解析成功: {n_ok}")
    print(f"解析失败: {n_fail}")
    if n_fail > 0:
        errors = defaultdict(int)
        for r in records:
            if not r.parse_ok:
                errors[r.parse_error or "unknown"] += 1
        for err, cnt in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  - {err}: {cnt}")

    # ── 2. RoleType 指标 ──
    print("\n>>> RoleType 评估")
    rt_valid = [(ROLE_CODE_TO_NAME.get(r.gt_role, r.gt_role),
                 ROLE_CODE_TO_NAME.get(r.pred_role, r.pred_role))
                for r in records
                if r.parse_ok and r.gt_role and r.pred_role
                and r.gt_role not in IGNORED_CATEGORIES["role_type"]]
    if rt_valid:
        gt_rt, pred_rt = zip(*rt_valid)
        acc_rt, stats_rt = compute_metrics(list(gt_rt), list(pred_rt))
        print(f"总体准确率: {acc_rt:.4f} ({acc_rt * 100:.2f}%)")
        print(build_table(stats_rt, ROLE_LABELS, "RoleType"))
        save_confusion_matrix(
            list(gt_rt), list(pred_rt),
            ROLE_LABELS,
            os.path.join(out_dir, "cm_role.png"),
            "Confusion Matrix - RoleType",
        )
    else:
        acc_rt = 0.0
        print("无有效 RoleType 数据")

    # ── 3. Sub-Keyscene 指标 ──
    print("\n>>> Sub-Keyscene 评估")
    sk_valid = [(SK_CODE_TO_NAME.get(r.gt_sk, r.gt_sk),
                 SK_CODE_TO_NAME.get(r.pred_sk, r.pred_sk))
                for r in records
                if r.parse_ok and r.gt_sk and r.pred_sk
                and r.gt_sk not in IGNORED_CATEGORIES["sub_keyscene"]]
    if sk_valid:
        gt_sk, pred_sk = zip(*sk_valid)
        acc_sk, stats_sk = compute_metrics(list(gt_sk), list(pred_sk))
        print(f"总体准确率: {acc_sk:.4f} ({acc_sk * 100:.2f}%)")
        print(build_table(stats_sk, SK_LABELS, "Sub-Keyscene"))
        save_confusion_matrix(
            list(gt_sk), list(pred_sk),
            SK_LABELS,
            os.path.join(out_dir, "cm_sub_keyscene.png"),
            "Confusion Matrix - Sub-Keyscene",
        )
    else:
        acc_sk = 0.0
        print("无有效 Sub-Keyscene 数据")

    # ── 4. KeyScene 指标 (从 SubKS 聚合) ──
    print("\n>>> KeyScene 评估 (从 Sub-Keyscene 聚合)")
    ks_valid = [
        (SK_TO_KEY_MAPPING.get(r.gt_sk), SK_TO_KEY_MAPPING.get(r.pred_sk))
        for r in records
        if r.parse_ok and r.gt_sk and r.pred_sk
        and r.gt_sk in SK_TO_KEY_MAPPING and r.pred_sk in SK_TO_KEY_MAPPING
    ]
    if ks_valid:
        gt_ks, pred_ks = zip(*ks_valid)
        acc_ks, stats_ks = compute_metrics(list(gt_ks), list(pred_ks))
        print(f"总体准确率: {acc_ks:.4f} ({acc_ks * 100:.2f}%)")
        print(build_table(stats_ks, KEYSCENE_LABELS, "KeyScene"))
        save_confusion_matrix(
            list(gt_ks), list(pred_ks), KEYSCENE_LABELS,
            os.path.join(out_dir, "cm_keyscene.png"),
            "Confusion Matrix - KeyScene",
        )
    else:
        acc_ks = 0.0
        print("无有效 KeyScene 数据")

    # ── 5. 汇总 ──
    report = {
        "n_total": len(records),
        "n_parse_ok": n_ok,
        "n_parse_fail": n_fail,
        "RoleType_acc": round(acc_rt, 4),
        "SubKeyScene_acc": round(acc_sk, 4),
        "KeyScene_acc": round(acc_ks, 4),
    }
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("评测完成")
    print(f"  RoleType:    {acc_rt:.4f} ({acc_rt * 100:.2f}%)")
    print(f"  SubKeyScene: {acc_sk:.4f} ({acc_sk * 100:.2f}%)")
    print(f"  KeyScene:    {acc_ks:.4f} ({acc_ks * 100:.2f}%)")
    print(f"  报告: {report_path}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(
        description="Anker VLM 预测准确率评测")
    ap.add_argument("--pred", required=True,
                    help="预测文件 (jsonl, 每行 {video_id, output})")
    ap.add_argument("--gt", required=True,
                    help="GT 文件 (jsonl, 每行 {video_id, labels})")
    ap.add_argument("--out-dir", default="eval_results",
                    help="输出目录 (默认: eval_results/)")
    ap.add_argument("--ignore-role", nargs="*", default=[],
                    help="屏蔽的 RoleType (代码, 如 C D)")
    ap.add_argument("--ignore-sk", nargs="*", default=[],
                    help="屏蔽的 Sub-Keyscene (代码, 如 g h)")
    ap.add_argument("--ignore-ks", nargs="*", default=[],
                    help="屏蔽的 KeyScene (全名, 如 'Normal Activity')")
    args = ap.parse_args()

    if args.ignore_role:
        # 支持代码或全名; 统一转为全名
        IGNORED_CATEGORIES["role_type"] = [
            ROLE_CODE_TO_NAME.get(v, v) for v in args.ignore_role]
    if args.ignore_sk:
        IGNORED_CATEGORIES["sub_keyscene"] = [
            SK_CODE_TO_NAME.get(v, v) for v in args.ignore_sk]
    if args.ignore_ks:
        IGNORED_CATEGORIES["keyscene"] = list(args.ignore_ks)

    run_evaluation(args.pred, args.gt, args.out_dir)


if __name__ == "__main__":
    main()