"""错标嫌疑抽查拼图: mislabel_suspects 抽样 → 16 帧 4×4 拼图 JPG。

  python3 jax_impl/spot_check_collage.py \
      --suspects outputs/cross_exam/mislabel_suspects.jsonl \
      --labels /data/labels_dedup.jsonl --out-dir outputs/cross_exam/imgs \
      [--n 20] [--exclude-classes mojs]

无可视化环境的抽查方案: 拼图 scp 回本地由人(或视觉模型)判
"GT 对还是双证人对"。文件名自带三方答案,看图即判,无需对表。
需在容器内跑(读分片 + PIL)。
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suspects", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--exclude-classes", default="mojs",
                    help="不抽的 GT 类(默认 m 豁免类 + o/j/s 帧间证据类)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    from PIL import Image
    from jax_impl.data import load_frames

    sus = [json.loads(l) for l in open(a.suspects, encoding="utf-8")]
    sus = [s for s in sus if s["gt"]["sk"] not in set(a.exclude_classes)]
    rng = random.Random(a.seed)
    rng.shuffle(sus)
    # 重灾区优先(e/l/g),再补随机
    picks = ([s for s in sus if s["gt"]["sk"] in "elg"][: a.n * 2 // 3]
             + sus)[: a.n]
    seen, uniq = set(), []
    for s in picks + sus:
        if s["video_id"] not in seen:
            seen.add(s["video_id"])
            uniq.append(s)
        if len(uniq) >= a.n:
            break

    recs = {}
    for line in open(a.labels, encoding="utf-8"):
        d = json.loads(line)
        recs.setdefault(d["video_id"], d)

    os.makedirs(a.out_dir, exist_ok=True)
    n_ok = 0
    for i, s in enumerate(uniq):
        rec = recs.get(s["video_id"])
        if rec is None:
            continue
        try:
            fr = load_frames(rec, os.path.dirname(a.labels))
        except Exception as e:  # noqa: BLE001
            print("  帧读取失败", s["video_id"], e)
            continue
        g = np.zeros((4 * 192, 4 * 192, 3), np.uint8)
        for k, f_ in enumerate(fr[:16]):
            im = np.asarray(Image.fromarray(f_).resize((192, 192)))
            g[(k // 4) * 192:(k // 4 + 1) * 192,
              (k % 4) * 192:(k % 4 + 1) * 192] = im
        name = (f"{i:02d}_GT-{s['gt']['sk']}_model-{s['model']['sk']}"
                f"_gemini-{s['gemini']['sk']}.jpg")
        Image.fromarray(g).save(os.path.join(a.out_dir, name),
                                "JPEG", quality=85)
        n_ok += 1
    print(f"[OK] {n_ok} 张拼图 → {a.out_dir}/(文件名 = 三方答案;"
          f"scp 回本地看图判'GT 对还是 model/gemini 对')")


if __name__ == "__main__":
    main()
