"""测试集失分决算审计: 全量错误逐条归因 → 六桶决算表(优化方向决策件)。

  python3 jax_impl/attribute_errors.py \
      --labels /data/labels_test.jsonl \
      --preds outputs/prior_recal/all.jsonl \
      --blind /data/test_blind.jsonl \
      [--preds-bare outputs/jax_5b_v4replica/eval_preds.jsonl]

输入:
  --preds       主口径预测(建议用交付口径=挂先验版;对它的错误做归因)
  --blind       Gemini 对测试视频的盲判(label_euno_wds 产出;纯预测,
                不触碰测试标签,合规)
  --preds-bare  可选,裸分口径预测 → 顺带量化先验外挂修了/引入多少

六桶(互斥,按优先级归类;每桶给"条数/折分/处方"):
  A 考卷重嫌   模型与盲判一致反对 GT —— 无药,属于天花板本身
  B 帧证据缺失 GT∈{o,j,s}(证据在 16 帧之间的类)—— 药=重切更多帧
  C 身份信息   h↔n 互混(认人问题)—— 药=扩数据/身份专项
  E 模型缺陷   盲判支持 GT 而模型错 —— 药=训练侧(S5/清洗/RL),
               唯一"可训练残差"
  F 真难样本   双证人都错且答案不同 —— 接近无药
  G 证人缺席   盲判无该样本记录 —— 补标后重跑
纯 stdlib。
"""
import argparse
import collections
import json

FRAME_BLIND = set("ojs")          # 帧间证据类(训练侧 unsupported 指纹佐证)
IDENTITY_PAIRS = {("h", "n"), ("n", "h")}


def load_preds(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        seg = (d.get("output") or "").split("|")
        if len(seg) >= 2:
            out[d["video_id"]] = (seg[0].strip(), seg[1].strip())
    return out


def load_blind(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        g = (d.get("gemini_output") or d).get("predictions") or {}
        if g.get("sub_keyscene") and d["video_id"] not in out:
            out[d["video_id"]] = (g.get("role_type"), g["sub_keyscene"])
    return out


def bucket_of(gt_sk, md_sk, bl):
    if bl is None:
        return "G_证人缺席"
    bl_sk = bl[1]
    if bl_sk == md_sk and bl_sk != gt_sk:
        return "A_考卷重嫌"
    if gt_sk in FRAME_BLIND:
        return "B_帧证据缺失"
    if (gt_sk, md_sk) in IDENTITY_PAIRS:
        return "C_身份信息"
    if bl_sk == gt_sk:
        return "E_模型缺陷"
    return "F_真难样本"


PRESCRIPTION = {
    "A_考卷重嫌": "无药(考卷噪声=天花板);可作测试集质量证据",
    "B_帧证据缺失": "重切 WDS(24~32 帧,客户侧管线)",
    "C_身份信息": "扩数据(人物多样性)/身份专项",
    "E_模型缺陷": "训练侧: S5 CoT/方向级改标/RL —— 唯一可训练残差",
    "F_真难样本": "接近无药(边界模糊)",
    "G_证人缺席": "补盲判后重跑本脚本",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--preds", required=True, help="主口径(建议交付口径)")
    ap.add_argument("--blind", required=True)
    ap.add_argument("--preds-bare", default=None)
    a = ap.parse_args()

    gt = {}
    for line in open(a.labels, encoding="utf-8"):
        d = json.loads(line)
        lab = d.get("labels") or d
        gt[d["video_id"]] = (lab["role_type"], lab["sub_keyscene"])
    md = load_preds(a.preds)
    bl = load_blind(a.blind)
    n_all = len(gt)

    buckets = collections.Counter()
    per_class = collections.defaultdict(collections.Counter)
    errors = 0
    for v, (g_rt, g_sk) in gt.items():
        m = md.get(v)
        if m is None or m[1] != g_sk:
            errors += 1
            b = bucket_of(g_sk, m[1] if m else "?", bl.get(v))
            buckets[b] += 1
            per_class[b][g_sk] += 1

    print(f"样本 {n_all} | 主口径 SubKS 错误 {errors} 条"
          f"(acc={1-errors/n_all:.2%})\n")
    print(f"{'桶':<14}{'条数':>7}{'占错误':>9}{'折分':>8}  处方")
    order = ["A_考卷重嫌", "B_帧证据缺失", "C_身份信息",
             "E_模型缺陷", "F_真难样本", "G_证人缺席"]
    for b in order:
        n = buckets[b]
        print(f"{b:<14}{n:>7}{n/max(errors,1):>9.1%}"
              f"{n/n_all*100:>7.2f}分  {PRESCRIPTION[b]}")
        top = per_class[b].most_common(4)
        if top:
            print(" " * 14 + "主要类: "
                  + " ".join(f"{k}={v}" for k, v in top))

    # 天花板估计: 考卷重嫌若全为真错标,理论上限 ≈ 100% − A桶折分 − F桶一半
    a_pts = buckets["A_考卷重嫌"] / n_all * 100
    f_pts = buckets["F_真难样本"] / n_all * 100
    e_pts = buckets["E_模型缺陷"] / n_all * 100
    cur = (1 - errors / n_all) * 100
    print(f"\n== 决算 ==")
    print(f"当前口径: {cur:.2f}")
    print(f"训练侧可挖(E桶全额): +{e_pts:.2f} → 理想 {cur+e_pts:.2f}")
    print(f"估计天花板(扣 A桶 与 F桶/2): ~{100-a_pts-f_pts/2:.1f}")
    print(f"B桶(帧)+C桶(身份)= 数据侧升级可赎回的部分: "
          f"+{(buckets['B_帧证据缺失']+buckets['C_身份信息'])/n_all*100:.2f}")

    if a.preds_bare:
        bare = load_preds(a.preds_bare)
        fixed = broke = 0
        for v, (g_rt, g_sk) in gt.items():
            b_ok = bare.get(v, ("", ""))[1] == g_sk
            m_ok = md.get(v, ("", ""))[1] == g_sk
            fixed += (not b_ok) and m_ok
            broke += b_ok and (not m_ok)
        print(f"\n先验外挂账目: 修好 {fixed} 条 / 打破 {broke} 条 "
              f"(净 {(fixed-broke)/n_all*100:+.2f} 分)")


if __name__ == "__main__":
    main()
