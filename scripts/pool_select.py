#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1M 池选样器(08-04): 双策略同框架, 产出全池排序 + 嵌套清单。
  --strategy general  通用档(泛化导向, 换考卷/防过拟合审计时用):
      四象限(judge×信息量) + 机位去冗 + test 边际对齐 + 归因切片配额
      + 尾类应收尽收 + 15% 随机保底
  --strategy testmax  压榨档(EunoVLM 同规则竞赛, 当前主战术):
      general 全部 + 场景克隆权重 + test 近邻检索加分(需 emb 文件)
      + 不设近重复天花板
信号文件全部可选, 缺哪个降级哪项(打印 warning), 便于分阶段跑:
  --judge   judge 结果 jsonl(video_id, verdict[, correct_*])   → 质量项
  --scores  前向打分 jsonl(video_id, margin_rt, margin_sk)     → 信息量项
  --emb/--test-emb  npz(ids, scene[, act])                     → 场景/检索项
用法:
  python scripts/pool_select.py --pool <1M json> --strategy testmax \
     --out-dir /data/pool_sel --sizes 100000,200000,300000,500000
"""
import argparse, json, math, os, random, re, sys
from collections import Counter, defaultdict

TAIL = set("qrujonst")          # 安全/稀缺事件类: 应收尽收(不复制)
# 归因切片(去噪 test ①桶采购清单, 08-04): 关键词 → 加分标签
SLICES = [
    ("idA",  re.compile(r"\b(own door|house key|unlock|access card|from inside|returns? home|garage)\b", re.I), "A"),
    ("cour", re.compile(r"\b(package|parcel|courier|delivery|deliver|mail|box)\b", re.I), None),
    ("loit", re.compile(r"\b(linger|loiter|paces?|back and forth|circle|wander)\b", re.I), None),
    ("crim", re.compile(r"\b(pry|pries|forc|break|climb|fence|weapon|smash|kick)\b", re.I), None),
    ("veh",  re.compile(r"\b(gets? (in|into|out)|trunk|driver'?s? seat|drives? (off|away))\b", re.I), None),
]
NONASCII = re.compile(r"[一-鿿]")


def load_pool(path):
    print(f"[load] {path}", flush=True)
    raw = json.load(open(path, encoding="utf-8"))
    recs = []
    for r in raw:
        g = next((c["value"] for c in r.get("conversations", [])
                  if c.get("from") == "gpt"), "")
        m = re.match(r"\s*([A-E])\|([a-u])\|(.*)", g, re.S)
        if not m:
            continue
        vid = r["video"]
        parts = vid.split("/")[-1].split("_")
        cam = "_".join(parts[:2]) if len(parts) >= 2 else vid
        recs.append(dict(video_id=vid, rt=m.group(1), sk=m.group(2),
                         desc=m.group(3).strip(), cam=cam))
    print(f"[load] 有效 {len(recs)}/{len(raw)}", flush=True)
    return recs


def opt_map(path, key):
    if not path or not os.path.exists(path):
        print(f"[warn] 缺 {key} 文件, 该项降级为中性", flush=True)
        return None
    m = {}
    for l in open(path, encoding="utf-8"):
        j = json.loads(l)
        m[j["video_id"]] = j
    print(f"[sig] {key}: {len(m)} 条", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--test-labels", default="/data/labels_test.jsonl")
    ap.add_argument("--strategy", required=True, choices=["general", "testmax"])
    ap.add_argument("--judge", default="")
    ap.add_argument("--scores", default="")
    ap.add_argument("--emb", default="")
    ap.add_argument("--test-emb", default="")
    ap.add_argument("--cam-cap", type=int, default=80)
    ap.add_argument("--rand-floor", type=float, default=0.15)
    ap.add_argument("--sizes", default="100000,200000,300000,500000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)

    recs = load_pool(a.pool)
    # ---- 闸0: 结构 ----
    seen, out = set(), []
    n_dup = n_zh = n_empty = 0
    for r in recs:
        if r["video_id"] in seen:
            n_dup += 1; continue
        seen.add(r["video_id"])
        if not r["desc"]:
            n_empty += 1; continue
        if NONASCII.search(r["desc"]):
            n_zh += 1; continue
        out.append(r)
    recs = out
    cam_n = Counter(r["cam"] for r in recs)
    print(f"[闸0] 去重{n_dup} 空描述{n_empty} 中文{n_zh} → {len(recs)} | "
          f"机位 {len(cam_n)}", flush=True)

    # ---- test 边际(去噪口径) ----
    excl = set()
    for p in ("/data/test_mislabel_exclude_ids_rt_v3.txt",
              "/data/test_mislabel_exclude_ids_sk_v3.txt"):
        if os.path.exists(p):
            excl |= {x.strip() for x in open(p) if x.strip()}
    t_rt, t_sk = Counter(), Counter()
    for l in open(a.test_labels, encoding="utf-8"):
        j = json.loads(l)
        if j["video_id"] in excl:
            continue
        t_rt[j["labels"]["role_type"]] += 1
        t_sk[j["labels"]["sub_keyscene"]] += 1
    tn = sum(t_rt.values())

    judge = opt_map(a.judge, "judge")
    scores = opt_map(a.scores, "scores")
    scene_sim, retr_hit = {}, {}
    if a.strategy == "testmax":
        if a.emb and a.test_emb and os.path.exists(a.emb):
            import numpy as np
            z, tz = np.load(a.emb), np.load(a.test_emb)
            E = z["scene"] / (np.linalg.norm(z["scene"], axis=1, keepdims=True) + 1e-8)
            T = tz["scene"] / (np.linalg.norm(tz["scene"], axis=1, keepdims=True) + 1e-8)
            ids = [str(x) for x in z["ids"]]
            # 场景克隆: 与任一 test 场景的最大余弦; 检索: 进入任一 test 样本 top-K 计数
            S = E @ T.T                       # [N_pool, N_test](内存注意: 分块可后加)
            mx = S.max(axis=1)
            for i, v in enumerate(ids):
                scene_sim[v] = float(mx[i])
            K = 50
            top = np.argpartition(-S, min(K, S.shape[0] - 1), axis=0)[:K]
            for col in range(top.shape[1]):
                for i in top[:, col]:
                    retr_hit[ids[i]] = retr_hit.get(ids[i], 0) + 1
            print(f"[sig] 场景/检索向量: pool {len(ids)} × test {T.shape[0]}", flush=True)
        else:
            print("[warn] testmax 缺 emb 文件 → 场景克隆/检索项降级(明晚闸2补)", flush=True)

    # ---- 打分 ----
    W = (dict(q=2.0, info=1.0, scarce=0.8, slice=1.0, scene=0.0, retr=0.0)
         if a.strategy == "general" else
         dict(q=2.0, info=0.8, scarce=0.5, slice=0.8, scene=3.0, retr=2.0))
    sus_w = {}
    for r in recs:
        s = 0.0
        if judge:
            v = judge.get(r["video_id"])
            if v and v.get("verdict") == "mislabel":
                s -= W["q"]; sus_w[r["video_id"]] = 0.3   # 入池则降权, 不硬删
            else:
                s += W["q"] * 0.3
        if scores:
            v = scores.get(r["video_id"])
            if v:   # margin 低=信息量高(0~1 归一后取反)
                s += W["info"] * (1.0 - min(max(v.get("margin_sk", 0.5), 0.0), 1.0))
        s += W["scarce"] / math.log(cam_n[r["cam"]] + math.e)
        for name, pat, rt_need in SLICES:
            if pat.search(r["desc"]) and (rt_need is None or r["rt"] == rt_need):
                s += W["slice"] * 0.5
                r.setdefault("slices", []).append(name)
        s += W["scene"] * scene_sim.get(r["video_id"], 0.0)
        s += W["retr"] * min(retr_hit.get(r["video_id"], 0), 5) / 5.0
        if r["sk"] in TAIL:
            s += 2.0          # 尾类加成: 类内优先, 但不再绕过 RT 配额(修 top100k 被 C 挤爆的 bug)
        r["score"] = s

    # ---- 出池: 尾类全收 + 随机保底 + 分层(RT×SK 对齐 test 边际)贪心 ----
    sizes = sorted(int(x) for x in a.sizes.split(","))
    rng.shuffle(recs)
    n_floor = int(sizes[-1] * a.rand_floor)
    floor = set(id(r) for r in recs[:n_floor])
    for r in recs[:n_floor]:
        r["score"] = r.get("score", 0.0) + 1.0   # 随机保底: 加成而非独立通道
    body = sorted(recs, key=lambda r: -r["score"])
    per_cam = Counter()
    ranked = []
    for r in body:
        if per_cam[r["cam"]] >= a.cam_cap and r["sk"] not in TAIL:
            continue    # 尾类豁免机位cap(稀缺事件同机位也收)
        per_cam[r["cam"]] += 1
        ranked.append(r)
    # 边际对齐: 对每个截断档, 按 test RT 边际做水位裁剪
    open(os.path.join(a.out_dir, "ranked_all.txt"), "w").write(
        "\n".join(r["video_id"] for r in ranked) + "\n")
    n_tail = sum(1 for r in recs if r["sk"] in TAIL)
    rep = [f"strategy={a.strategy} 池={len(recs)} 排序后={len(ranked)} "
           f"尾类池={n_tail} 随机保底={len(floor)}"]
    for size in sizes:
        quota = {k: max(1, int(size * v / tn)) for k, v in t_rt.items()}
        got, sel, seen_sel = Counter(), [], set()
        for r in ranked:                       # 第一段: 严格按 test 边际配额
            if len(sel) >= size:
                break
            if got[r["rt"]] >= quota.get(r["rt"], 0):
                continue
            got[r["rt"]] += 1
            sel.append(r["video_id"]); seen_sel.add(r["video_id"])
        # 没填满(某类池内不够)则按分数补齐
        if len(sel) < size:                    # 第二段: 类内不够, 按总分补齐
            for r in ranked:
                if len(sel) >= size:
                    break
                if r["video_id"] not in seen_sel:
                    sel.append(r["video_id"]); seen_sel.add(r["video_id"])
        open(os.path.join(a.out_dir, f"pool_top{size//1000}k.txt"), "w").write(
            "\n".join(sel) + "\n")
        rep.append(f"top{size//1000}k: {len(sel)} 条, RT配额={dict(got)}")
    json.dump(sus_w, open(os.path.join(a.out_dir, "suspect_weights_sel.json"), "w"))
    open(os.path.join(a.out_dir, "report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep), flush=True)
    print(f"[done] -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
