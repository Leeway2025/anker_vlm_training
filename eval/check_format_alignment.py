"""交付前格式字节核对(training_plan 6.1:开训前必做,防低级翻车)。

背景: PDF 里的生产 prompt 代码块含换行断裂与续行符(疑似排版产物),
configs/prompt.txt 是清理版 —— 因此**整个输出格式的空格/换行约定**
必须与客户真实 GT 逐字节核对,不止 " | " 分隔符一处。

两项检查:
  ① GT 字节核对: 客户提供 3+ 条真实 GT 输出原始串,
     parse → 按 configs/base.yaml 的 separator 重建 → 与原串逐字节比对。
     不一致 = 空格约定不同 → 改 format.separator,并同步核对
     configs/prompt.txt 与生产 prompt 的空格/换行。
  ② tokenizer 检查: 分类字母在 Gemma tokenizer 里的切分是否稳定
     (formatting.char_spans_to_token_weights 用"区间最大"规则兜底,
      即使字母与分隔符合并成一个 token 也能拿到高权重,此处仅报告)。

用法:
  python -m eval.check_format_alignment --gt-samples gt_samples.txt \
      [--tokenizer google/gemma-4-e2b-it]
  gt_samples.txt: 每行一条真实 GT 输出原始串(从客户数据文件原样复制,
                  不要手敲 —— 手敲会丢失真实的空格约定)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.format_validator import parse_output          # noqa: E402
from data.formatting import build_target                # noqa: E402


def verify_gt_line(line: str, sep: str = "|"):
    """单条 GT 原始串核对。返回 None(逐字节一致)或错误说明字符串。"""
    r = parse_output(line)
    if not r.ok:
        return f"parse failed: {r.error}"
    if r.case_fixed:
        return (f"GT 里出现大小写异常({line!r})—— GT 本身不该需要矫正,"
                f"确认这条是不是真实 GT")
    rebuilt = build_target(r.rt, r.sk, r.desc, sep).text
    if rebuilt != line:
        return ("byte mismatch:\n"
                f"    original = {line!r}\n"
                f"    rebuilt  = {rebuilt!r}\n"
                "    → 调整 configs/base.yaml format.separator 后重跑本检查")
    return None


def tokenizer_report(tokenizer_name: str, sep: str = "|"):
    """②: 分类字母切分报告。返回 (字母, 上下文, token 串) 异常列表。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    issues = []
    letters = list("ABCDE") + list("abcdefghijklmnopqrstu")
    for L in letters:
        # RT 字母在行首、SK 字母跟在分隔符后 —— 两种真实上下文
        for ctx, text in (("line-start", f"{L}{sep}"),
                          ("after-sep", f"{sep}{L}{sep}")):
            ids = tok.encode(text, add_special_tokens=False)
            pieces = [tok.decode([i]) for i in ids]
            if not any(p.strip() == L for p in pieces):
                issues.append((L, ctx, pieces))
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-samples", required=True,
                    help="每行一条真实 GT 输出原始串")
    ap.add_argument("--tokenizer", default=None,
                    help="Gemma 4 模型/分词器路径(可选,做字母切分检查)")
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    sep = cfg["format"]["separator"]

    lines = [l.rstrip("\n") for l in
             open(a.gt_samples, encoding="utf-8") if l.strip()]
    if len(lines) < 3:
        print(f"[WARN] 只有 {len(lines)} 条样本,建议 ≥3 条")

    bad = 0
    for i, line in enumerate(lines, 1):
        err = verify_gt_line(line, sep)
        if err:
            bad += 1
            print(f"  FAIL #{i}: {err}")
        else:
            print(f"  OK   #{i}")
    print(f"\n[①字节核对] {len(lines) - bad}/{len(lines)} 一致 "
          f"(separator={sep!r})")

    if a.tokenizer:
        issues = tokenizer_report(a.tokenizer, sep)
        if issues:
            print(f"[②tokenizer] {len(issues)} 处字母与邻接字符合并成单 token"
                  f"(加权用区间最大规则兜底,不阻塞,但记录在案):")
            for L, ctx, pieces in issues:
                print(f"    {L!r} @ {ctx}: {pieces}")
        else:
            print("[②tokenizer] 全部分类字母落在独立 token 上 ✓")

    if bad:
        sys.exit(1)
    print("\n全部通过 —— 可以开训")


if __name__ == "__main__":
    main()
