#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闸0-lite: 抽帧是不是 RT/SubKS 的瓶颈? —— 纯 logits+GT 结构筛查,无需视觉模型。
逻辑: 重切帧只可能挽回"证据不足/真类是近次选"的错例;对"自信答错(真类被埋在第3+)"
      的错例,同一段视频换采样几乎不可能翻盘(那是感知/偏见,不是覆盖)。
      => 真类进 top-2 的错例占比 = 重切帧能挽回的【上限】,即 GPU 该不该动的 go/no-go。
局限: 这是【必要条件筛查】,非证明。"自信答错"里极少数可能是"误导帧在场+关键帧漏采",
      只有 Phase B(客户导稠密帧 + SigLIP2 身份打分)能进一步区分。全表口径: RT 与 SubKS 都看。
用法: python3 scripts/diag_frame_bottleneck.py [--preds outputs/optin/preds.jsonl]
                                              [--labels /data/labels_test.jsonl]
"""
import argparse, json, math

RT_SET = "ABCDE"
SK_SET = "abcdefghijklmnopqrstu"


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def rank_of(probs, gi):
    """真类 gi 在概率降序里的名次(1=最高)。"""
    p = probs[gi]
    return 1 + sum(1 for q in probs if q > p)


def load_labels(path):
    gt = {}
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        lb = d.get("labels") or d
        gt[d["video_id"]] = (lb["role_type"], lb["sub_keyscene"])
    return gt


def analyze(name, SET, logit_key, preds, gt, topk_cand=2):
    """返回 (acc, 错例总数, 可救上限数, A↔D式硬混淆信息 via caller)。"""
    n = correct = 0
    errs = []          # (vid, pred_idx, gt_idx, p_pred, p_gt, rank_gt, probs)
    conf = {}          # 混淆计数 (gt_letter, pred_letter) -> n
    for vid, rec in preds.items():
        if vid not in gt or logit_key not in rec:
            continue
        g = gt[vid][0 if logit_key == "rt" else 1]
        if g not in SET:
            continue
        gi = SET.index(g)
        probs = softmax(rec[logit_key])
        pi = max(range(len(probs)), key=lambda i: probs[i])
        n += 1
        if pi == gi:
            correct += 1
            continue
        conf[(g, SET[pi])] = conf.get((g, SET[pi]), 0) + 1
        errs.append((vid, pi, gi, probs[pi], probs[gi], rank_of(probs, gi), probs))
    acc = correct / max(n, 1)

    # 错例按"真类名次"分层
    rank_hist = {}
    for e in errs:
        rank_hist[e[5]] = rank_hist.get(e[5], 0) + 1
    cand = [e for e in errs if e[5] <= topk_cand]          # 真类进 top-k → 可救候选
    buried = [e for e in errs if e[5] > topk_cand]         # 真类被埋 → 重切帧无望
    # 置信度视角: 预测类概率(答得多"死")
    def avg(xs): return sum(xs) / max(len(xs), 1)
    conf_wrong = [e for e in errs if e[3] >= 0.60 and e[5] >= 3]  # 又自信又把真类埋了

    print(f"\n===== {name} =====")
    print(f"  样本 {n} | 正确 {correct} | acc = {100*acc:.2f}%")
    print(f"  错例 {len(errs)} 条,按【真类名次】分层:")
    for r in sorted(rank_hist):
        bar = "#" * round(40 * rank_hist[r] / max(len(errs), 1))
        print(f"    真类第{r}名: {rank_hist[r]:4d} ({100*rank_hist[r]/max(len(errs),1):4.1f}%) {bar}")
    print(f"  ── 可救上限(真类∈top-{topk_cand}): {len(cand)} 条 "
          f"= 若全翻盘 acc +{100*len(cand)/max(n,1):.2f} → {100*(correct+len(cand))/max(n,1):.2f}%")
    print(f"  ── 重切帧无望(真类被埋第{topk_cand+1}+): {len(buried)} 条 "
          f"({100*len(buried)/max(len(errs),1):.1f}% 的错例)")
    print(f"  ── 其中【自信答错】(预测类p≥0.60 且真类埋第3+): {len(conf_wrong)} 条 "
          f"→ 感知/偏见,非采样")
    print(f"     候选错例平均: 预测类p={avg([e[3] for e in cand]):.2f} 真类p={avg([e[4] for e in cand]):.2f}"
          f"(差越小越易被证据推翻)")
    return acc, errs, conf, cand, buried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="outputs/optin/preds.jsonl")
    ap.add_argument("--labels", default="/data/labels_test.jsonl")
    args = ap.parse_args()

    gt = load_labels(args.labels)
    preds = {}
    for l in open(args.preds, encoding="utf-8"):
        d = json.loads(l)
        preds[d["video_id"]] = d
    print(f"[in] preds={args.preds} ({len(preds)}) | labels={args.labels} ({len(gt)})")

    _, rt_errs, rt_conf, rt_cand, rt_bur = analyze("RoleType (身份)", RT_SET, "rt", preds, gt, topk_cand=2)
    # RT 混淆 top 榜(A↔D 是病根)
    print("\n  RT 混淆 top6 (真→预测):")
    for (g, p), c in sorted(rt_conf.items(), key=lambda x: -x[1])[:6]:
        print(f"    {g}→{p}: {c}")
    # A↔D 专项: 这些错例的可救性
    ad = [e for e in rt_errs if {RT_SET[e[1]], RT_SET[e[2]]} == {"A", "D"}]
    if ad:
        ad_cand = sum(1 for e in ad if e[5] <= 2)
        print(f"  A↔D 混淆 {len(ad)} 条: 真类∈top2 的 {ad_cand} 条 "
              f"({100*ad_cand/len(ad):.0f}%) 可救;其余为硬混淆")

    analyze("SubKS (主指标)", SK_SET, "sk", preds, gt, topk_cand=2)

    print("\n===== 判读 =====")
    print("  · '可救上限' 是乐观天花板(假设候选全翻盘);实际重切帧只能兑现其中一部分。")
    print("  · 若 RT 可救上限很小 / '重切帧无望' 占多数 → 抽帧不是主瓶颈,GPU 重切性价比低,")
    print("    该把弹药投到感知(更干净标注 / CoT)或集成,而非重切帧(参见 KTO 拉错杠杆教训)。")
    print("  · 若可救上限可观、且候选里真类p 接近预测类p → 抽帧是真杠杆,进 Phase B:")
    print("    客户导'候选错例'的稠密帧 → SigLIP2 身份打分 → 确认稠密里存在 uniform-16 漏掉的身份帧。")


if __name__ == "__main__":
    main()
