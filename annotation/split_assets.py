"""标注产出 → 训练资产文件(Step 2 与 Step 5/6 之间的衔接件)。

输入: gemini 标注 jsonl(label_euno_wds / gemini_labeler 产出,
  每行 {"video_id", "gemini_output": {attributes, reasoning_chain, ...}})
输出(即 configs/base.yaml data.* 指向的三个文件):
  asset_A_attributes.jsonl  {"video_id", "confidence", "attributes"}  → Phase 5b
  asset_C_reasoning.jsonl   {"video_id", "reasoning_chain"}           → Phase 5d
  asset_D_whitelist.txt     (可选 --whitelist 原样拷入,统一落位)

⚠️ 没有这一步,直接把 pass1.jsonl 喂训练会**静默拿不到辅助头标签**
(训练读的是顶层 attributes 字段,而标注文件嵌在 gemini_output 里)。

用法:
  python -m annotation.split_assets --gemini pass1.jsonl \
      [--whitelist filtered/whitelist_ids.txt] --out-dir DATA/
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def split(gemini_path, out_dir, whitelist_path=None):
    """传 whitelist 时,资产文件**只写白名单内的条目** —— 过滤在资产层
    完成,训练样本永远全量(客户 GT 质量高于 Gemini,白名单只约束
    Gemini 自产资产的采用范围,绝不裁样本/裁 GT)。"""
    wl = set(open(whitelist_path).read().split()) if whitelist_path else None
    os.makedirs(out_dir, exist_ok=True)
    n = n_bad = n_out_wl = 0
    with open(os.path.join(out_dir, "asset_A_attributes.jsonl"), "w",
              encoding="utf-8") as fa, \
            open(os.path.join(out_dir, "asset_C_reasoning.jsonl"), "w",
                 encoding="utf-8") as fc:
        for line in open(gemini_path, encoding="utf-8"):
            d = json.loads(line)
            if wl is not None and d["video_id"] not in wl:
                n_out_wl += 1        # Gemini 与 GT 不一致 → 其资产不采用
                continue
            g = d.get("gemini_output") or {}
            attrs, chain = g.get("attributes"), g.get("reasoning_chain")
            if not attrs or not chain:
                n_bad += 1
                continue
            # 记录已过 validate_record 才落盘(labeler 保证)→ confidence 1.0
            fa.write(json.dumps({"video_id": d["video_id"],
                                 "confidence": 1.0, "attributes": attrs},
                                ensure_ascii=False) + "\n")
            fc.write(json.dumps({"video_id": d["video_id"],
                                 "reasoning_chain": chain},
                                ensure_ascii=False) + "\n")
            n += 1
    if whitelist_path:
        shutil.copy(whitelist_path,
                    os.path.join(out_dir, "asset_D_whitelist.txt"))
    print(f"{n} 条 → asset_A/asset_C(缺字段 {n_bad},白名单外 {n_out_wl}"
          f" —— 这些样本仍全量参训,只是无增强监督)"
          f"{';whitelist 已拷入' if whitelist_path else ''} -> {out_dir}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini", required=True, help="pass1.jsonl")
    ap.add_argument("--whitelist", default=None,
                    help="consistency_filter 产出的 whitelist_ids.txt")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    split(a.gemini, a.out_dir, a.whitelist)


if __name__ == "__main__":
    main()
