"""标注口径盲审 —— 工作包生成器。

  python3 jax_impl/audit_sample.py --preds outputs/jax_5b_v2/eval_preds.jsonl \
      --labels DATA/labels_test.jsonl --out-dir outputs/audit \
      [--frames]        # 附带每条 16 帧拼图(需分片可读,容器内跑)

产出:
  worksheet.csv   审计表(候选1/候选2 已随机盲化,人工填"判定"列:
                  1 / 2 / both / neither / unsure)
  answer_key.json 盲化密钥(审完前勿看)
  imgs/*.jpg      16 帧拼图(--frames 时)
分层抽样(v2 失分解剖的四个疑点):
  m_disputed   gt=m 而模型判具体类(懒标嫌疑主战场)  30 条
  s_boundary   s(Loitering)相关分歧                 20 条
  rt_AD        RT 的 A↔D 分歧                        20 条
  control      模型与 GT 一致的对照组(校准审计者)    10 条
"""
import argparse
import collections
import csv
import json
import os
import random
import sys

# 仓库根优先于脚本目录 —— 否则 jax_impl/data.py 会遮蔽根目录 data 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.taxonomy import RT_NAMES  # noqa: E402

SK_NAMES = {
    "a": "Vehicle Access", "b": "Dog Walking", "c": "Kid Playing",
    "d": "Kid Studying", "e": "Leisure Activity", "f": "Home Chores",
    "g": "Visitor Arrival", "h": "Package Brought Home",
    "i": "Package Delivery", "j": "Person Falling", "k": "Leaving Porch",
    "l": "Approaching Porch", "m": "Other Normal Activity",
    "n": "Package Taken Away", "o": "Other Property Damage",
    "p": "Wildlife", "q": "Weapon Threat", "r": "Other Hazards",
    "s": "Loitering", "t": "Vehicle Anomaly", "u": "Unauthorized Entry",
}


def show(rt, sk, desc):
    return (f"{rt}|{sk}  [{RT_NAMES.get(rt, '?')} / {SK_NAMES.get(sk, '?')}]"
            f"  desc: {desc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", action="store_true",
                    help="导出 16 帧拼图(需能读分片,建议容器内跑)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    preds = {j["video_id"]: j["output"] for j in
             map(json.loads, open(a.preds, encoding="utf-8"))}
    gt = [json.loads(l) for l in open(a.labels, encoding="utf-8")]

    def parsed(vid):
        seg = preds.get(vid, "").split("|")
        if len(seg) < 3:
            return None
        return seg[0].strip(), seg[1].strip(), "|".join(seg[2:]).strip()

    strata = {"m_disputed": [], "s_boundary": [], "rt_AD": [], "control": []}
    for j in gt:
        p = parsed(j["video_id"])
        if p is None:
            continue
        prt, psk, pdesc = p
        grt, gsk = j["labels"]["role_type"], j["labels"]["sub_keyscene"]
        item = (j, prt, psk, pdesc)
        if gsk == "m" and psk != "m":
            strata["m_disputed"].append(item)
        elif (gsk == "s") != (psk == "s") and (gsk == "s" or psk == "s"):
            strata["s_boundary"].append(item)
        elif {grt, prt} == {"A", "D"} and gsk == psk:
            strata["rt_AD"].append(item)
        elif grt == prt and gsk == psk:
            strata["control"].append(item)
    quota = {"m_disputed": 30, "s_boundary": 20, "rt_AD": 20, "control": 10}
    rows, key = [], {}
    for name, items in strata.items():
        rng.shuffle(items)
        for j, prt, psk, pdesc in items[: quota[name]]:
            vid = j["video_id"]
            gt_txt = show(j["labels"]["role_type"], j["labels"]["sub_keyscene"],
                          j["labels"]["description"])
            md_txt = show(prt, psk, pdesc)
            gt_first = rng.random() < 0.5
            c1, c2 = (gt_txt, md_txt) if gt_first else (md_txt, gt_txt)
            idx = len(rows) + 1
            rows.append({"idx": idx, "video_id": vid,
                         "img": f"imgs/{vid.replace('/', '__')}.jpg",
                         "候选1": c1, "候选2": c2, "判定(1/2/both/neither/unsure)": ""})
            key[str(idx)] = {"stratum": name, "gt_is": 1 if gt_first else 2,
                             "video_id": vid}
    with open(os.path.join(a.out_dir, "worksheet.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(key, open(os.path.join(a.out_dir, "answer_key.json"), "w",
                        encoding="utf-8"))
    print("分层抽样:", {k: min(len(v), quota[k]) for k, v in strata.items()},
          f"→ {len(rows)} 条")

    if a.frames:
        import numpy as np
        from PIL import Image
        from jax_impl.data import load_frames
        os.makedirs(os.path.join(a.out_dir, "imgs"), exist_ok=True)
        by_vid = {j["video_id"]: j for j in gt}
        for r in rows:
            j = by_vid[r["video_id"]]
            try:
                fr = load_frames(j, os.path.dirname(a.labels))
            except Exception as e:  # noqa: BLE001
                print("  帧读取失败", r["video_id"], e)
                continue
            g = np.zeros((4 * 192, 4 * 192, 3), np.uint8)
            for k, f_ in enumerate(fr[:16]):
                im = np.asarray(Image.fromarray(f_).resize((192, 192)))
                g[(k // 4) * 192:(k // 4 + 1) * 192,
                  (k % 4) * 192:(k % 4 + 1) * 192] = im
            Image.fromarray(g).save(
                os.path.join(a.out_dir, r["img"]), "JPEG", quality=85)
        print("拼图导出完成 →", os.path.join(a.out_dir, "imgs/"))
    print(f"[OK] 工作包 → {a.out_dir}/worksheet.csv(审完跑 audit_score.py)")


if __name__ == "__main__":
    main()
