# -*- coding: utf-8 -*-
"""可疑标注挖掘 —— 交叉盲判 × 多种子合议 × 逐类阈值(CL) × 近邻标签冲突。

产出"机器排队、人来定性"的复核队列(worksheet),以及复核后的纯度曲线
与截断点建议。定位:只排队,不改标 —— 定性权(真错标 vs 规则口径差)
永远在人(改判 522 条 → RT −1.8 的教训,规则口径差是学习目标不是噪声)。

① 排队(mine):
  python3 jax_impl/label_audit.py mine \
      --logits outputs/audit/seed1_foldout.jsonl \
      --logits outputs/audit/seed2_foldout.jsonl \
      --logits outputs/audit/seed3_foldout.jsonl \
      --labels /data/labels_train.jsonl \
      [--emb outputs/audit/emb.npz] [--knn 15] [--temp 1.0] \
      [--min-seeds 2] [--top 5000] \
      --out outputs/audit/suspects.csv

  · 每个 --logits 是一个种子的**折外**推理产物(infer.py --dump-letter-logits,
    行含 rt/sk 裸 logits)。红线:每条样本的 logits 必须来自没训过它的模型
    (A 折模型推 B 折)。折内自测会把错标背下来、信号整个反掉 —— 本工具
    无法检查这一点,由调用方保证。
  · --emb 可选:npz 含 "emb" [N,D](折外特征,倒二层池化),旁伴
    emb.ids.json 存 video_id 顺序(与 ext_score npz 同款约定)。给了就
    叠加近邻标签冲突信号,专抓同源池"近重复片段标注不一致"。

② 复核后定截断(purity):
  python3 jax_impl/label_audit.py purity \
      --worksheet outputs/audit/suspects_reviewed.csv \
      [--target 0.8] [--window 200]

  worksheet 的 verdict 列由人工填: mislabel / rule_ok / unsure(空=未看)。
  输出逐段纯度与"滚动纯度跌破 target 的排名"= 复核预算截断点。

信号与打分(全部落在输出列里,便于人工核对):
  margin      = log p(ŷ|x) − log p(y|x),折外+温度后,多种子取均值。
                双条件的连续化: 模型坚定押别处 且 坚定否定标签才得高分;
                0.5 vs 0.4 的摇摆(难例)天然低分。
  cl_votes    = 满足 confident-learning 判据的种子数。判据: 存在 c′≠y 使
                p(c′|x) ≥ t_c′,t_c′ = "标注为 c′ 的样本上 p(c′) 的均值"
                (逐类自适应阈值 —— 21 类失衡,全局阈值会让高频类淹没队列)。
  knn_conflict= 1 − 近邻同标签比例(cosine top-k);knn_suggest = 近邻多数标签。
  score       = relu(margin_mean) × (cl_votes/n_seeds) × (0.5 + 0.5×knn_conflict)
                (无 --emb 时第三项恒为 1)。

体温计:打印各任务折外盲判命中率。健康水位 ~0.37(SubKS);若 > 0.9
几乎必是标签泄漏混入(rate 0.99 前科),队列作废先查泄漏。纯 stdlib+numpy。
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

RT_SET = "ABCDE"
SK_SET = "abcdefghijklmnopqrstu"
TASKS = (("rt", RT_SET, "role_type"), ("sk", SK_SET, "sub_keyscene"))


# ---------------------------------------------------------------- 载入

def load_labels(path):
    lab = {}
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        v = d.get("labels") or d
        lab[d["video_id"]] = (v["role_type"], v["sub_keyscene"])
    return lab


def load_seed_logits(path, lab):
    """一个种子的折外 letter-logits → {task: {vid: np.ndarray}}。"""
    out = {"rt": {}, "sk": {}}
    n_skip = 0
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        vid = d.get("video_id")
        if vid not in lab or "rt" not in d or "sk" not in d:
            n_skip += 1
            continue
        out["rt"][vid] = np.asarray(d["rt"], np.float64)
        out["sk"][vid] = np.asarray(d["sk"], np.float64)
    if n_skip:
        print(f"[load] {os.path.basename(path)}: 跳过 {n_skip} 行(缺 logits 或无标签)")
    return out


def softmax(z, temp):
    z = z / max(temp, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- mine

def per_class_thresholds(P, y_idx, n_cls):
    """CL 逐类阈值 t_c = 标注为 c 的样本上 p(c) 的均值(自信度均值)。"""
    t = np.zeros(n_cls)
    for c in range(n_cls):
        m = y_idx == c
        # 该类无样本时置 1.0(不可能触发),避免除零
        t[c] = P[m, c].mean() if m.any() else 1.0
    return t


def knn_conflict(emb_npz, vids_order, y_by_vid, k, letters):
    """近邻标签冲突: 返回 {vid: (conflict, suggest_letter)}。cosine top-k,分块。"""
    z = np.load(emb_npz)
    ids_path = emb_npz.rsplit(".npz", 1)[0] + ".ids.json"
    ids = json.load(open(ids_path, encoding="utf-8"))
    E = np.asarray(z["emb"], np.float32)
    assert len(ids) == E.shape[0], f"ids {len(ids)} != emb {E.shape[0]}"
    keep = [i for i, v in enumerate(ids) if v in y_by_vid]
    ids = [ids[i] for i in keep]
    E = E[keep]
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    y = np.array([letters.index(y_by_vid[v]) for v in ids], np.int32)
    n = len(ids)
    out = {}
    B = 2048  # 分块避免 N×N 矩阵
    for s in range(0, n, B):
        sim = E[s:s + B] @ E.T                     # [b, n]
        for r in range(sim.shape[0]):
            sim[r, s + r] = -2.0                   # 去掉自己
        # top-k 近邻(argpartition O(n))
        nb = np.argpartition(-sim, kth=min(k, n - 1), axis=1)[:, :k]
        for r in range(sim.shape[0]):
            lbl = y[nb[r]]
            same = (lbl == y[s + r]).mean()
            maj = np.bincount(lbl, minlength=len(letters)).argmax()
            out[ids[s + r]] = (float(1.0 - same), letters[int(maj)])
    return out


def cmd_mine(a):
    lab = load_labels(a.labels)
    seeds = [load_seed_logits(p, lab) for p in a.logits]
    n_seeds = len(seeds)
    print(f"[mine] {n_seeds} 个种子折外 logits,标签 {len(lab)} 条")

    rows = []
    for task, letters, lab_key in TASKS:
        # 各种子共有的 vid 才可合议(缺席种子不硬凑)
        vids = sorted(set.intersection(*[set(s[task]) for s in seeds]))
        if not vids:
            print(f"[mine] task={task}: 无共有样本,跳过")
            continue
        y_idx = np.array([letters.index(lab[v][0 if task == "rt" else 1])
                          for v in vids], np.int32)
        n_cls = len(letters)

        margins = np.zeros((n_seeds, len(vids)))
        cl_flag = np.zeros((n_seeds, len(vids)), bool)
        pred_votes = np.zeros((len(vids), n_cls), np.int32)
        agree = np.zeros(n_seeds)
        for si, s in enumerate(seeds):
            L = np.stack([s[task][v] for v in vids])
            P = softmax(L, a.temp)
            top = P.argmax(axis=1)
            agree[si] = (top == y_idx).mean()
            logp = np.log(P + 1e-12)
            margins[si] = logp[np.arange(len(vids)), top] - \
                logp[np.arange(len(vids)), y_idx]
            t_c = per_class_thresholds(P, y_idx, n_cls)
            # CL 判据: 存在 c′≠y 使 p(c′) ≥ t_c′
            hit = P >= t_c[None, :]
            hit[np.arange(len(vids)), y_idx] = False
            cl_flag[si] = hit.any(axis=1) & (top != y_idx)
            pred_votes[np.arange(len(vids)), top] += 1

        # 体温计: 折外盲判命中率(泄漏指纹 0.99 / 健康 ~0.37)
        for si in range(n_seeds):
            tag = "⚠️ 疑似标签泄漏,队列作废先查泄漏!" if agree[si] > 0.9 else ""
            print(f"[盲判率] task={task} seed{si}: {agree[si]:.3f} {tag}")

        knn = {}
        if a.emb:
            y_by_vid = {v: lab[v][0 if task == "rt" else 1] for v in vids}
            knn = knn_conflict(a.emb, vids, y_by_vid, a.knn, letters)

        m_mean = margins.mean(axis=0)
        votes = cl_flag.sum(axis=0)
        for i, v in enumerate(vids):
            if votes[i] < a.min_seeds or m_mean[i] <= 0:
                continue
            conf, sug = knn.get(v, (None, None))
            k_term = 1.0 if conf is None else 0.5 + 0.5 * conf
            score = max(m_mean[i], 0.0) * (votes[i] / n_seeds) * k_term
            rows.append({
                "video_id": v, "task": task,
                "label": letters[y_idx[i]],
                "model_suggest": letters[int(pred_votes[i].argmax())],
                "score": round(float(score), 4),
                "margin_mean": round(float(m_mean[i]), 4),
                "cl_votes": f"{int(votes[i])}/{n_seeds}",
                "knn_conflict": "" if conf is None else round(conf, 3),
                "knn_suggest": sug or "",
                "verdict": "", "note": "",
            })

    rows.sort(key=lambda r: -r["score"])
    if a.top:
        rows = rows[: a.top]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cols = ["rank", "video_id", "task", "label", "model_suggest", "score",
            "margin_mean", "cl_votes", "knn_conflict", "knn_suggest",
            "verdict", "note"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            r["rank"] = i + 1
            w.writerow(r)
    by_task = collections.Counter(r["task"] for r in rows)
    print(f"[mine] 可疑队列 {len(rows)} 条 → {a.out}  {dict(by_task)}")
    print("[mine] 下一步: 人工填 verdict(mislabel/rule_ok/unsure)→ "
          "purity 子命令定复核截断点。规则口径差(rule_ok)一条不改!")


# ---------------------------------------------------------------- purity

def cmd_purity(a):
    rows = list(csv.DictReader(open(a.worksheet, encoding="utf-8")))
    rows.sort(key=lambda r: int(r["rank"]))
    seen = [r for r in rows if r.get("verdict", "").strip()]
    if not seen:
        raise SystemExit("worksheet 无已填 verdict 的行 —— 先人工复核再来")
    print(f"[purity] 已复核 {len(seen)}/{len(rows)} 条")
    hits = np.array([r["verdict"].strip() == "mislabel" for r in seen], float)
    cum = hits.cumsum() / (np.arange(len(hits)) + 1)
    w = min(a.window, len(hits))
    roll = np.convolve(hits, np.ones(w) / w, mode="valid")
    cut = None
    for i, p in enumerate(roll):
        if p < a.target:
            cut = int(seen[i + w - 1]["rank"])
            break
    step = max(len(seen) // 10, 1)
    print(f"{'排名':>6}  {'累计纯度':>8}  {'滚动纯度(w=%d)' % w}")
    for i in range(step - 1, len(seen), step):
        rp = roll[max(i - w + 1, 0)] if i >= w - 1 else float("nan")
        print(f"{seen[i]['rank']:>6}  {cum[i]:>8.3f}  {rp:>8.3f}")
    vc = collections.Counter(r["verdict"].strip() for r in seen)
    print(f"[purity] 判定分布: {dict(vc)}")
    if cut:
        print(f"[purity] 滚动纯度首次跌破 {a.target} @ rank {cut} → "
              f"建议复核预算截断于此;之后按 unsure 抽检即可")
    else:
        print(f"[purity] 已复核段滚动纯度未跌破 {a.target} → 队列还有肉,可继续")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine", help="产出可疑标注排序队列")
    m.add_argument("--logits", action="append", required=True,
                   help="折外 letter-logits jsonl,可重复(每种子一个)")
    m.add_argument("--labels", required=True)
    m.add_argument("--emb", help="折外特征 npz(键 emb)+ 旁伴 .ids.json")
    m.add_argument("--knn", type=int, default=15)
    m.add_argument("--temp", type=float, default=1.0,
                   help="softmax 温度(有校准温度就填,没有 1.0 也行: "
                        "CL 逐类阈值对未校准分布自适应)")
    m.add_argument("--min-seeds", type=int, default=2,
                   help="至少几个种子 CL 判据同亮才入队(k-of-n)")
    m.add_argument("--top", type=int, default=5000,
                   help="队列截断(性价比版预算 ~5k;0=不截)")
    m.add_argument("--out", required=True)
    p = sub.add_parser("purity", help="复核后画纯度曲线、定截断点")
    p.add_argument("--worksheet", required=True)
    p.add_argument("--target", type=float, default=0.8)
    p.add_argument("--window", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "mine":
        if a.top == 0:
            a.top = None
        cmd_mine(a)
    else:
        cmd_purity(a)


if __name__ == "__main__":
    main()
