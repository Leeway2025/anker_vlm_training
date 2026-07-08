"""代理数据集示例转换器: UCSD Anomaly(ped2)→ 客户 labels.jsonl 格式。

用途(dataset_plan 轨道 1 的最小演示):
  真实监控视角(固定俯拍人行道)的开放数据 → 按客户数据格式落盘,
  供数据管线/训练冒烟消费。标签为**规则伪标**(占位):
    Train 片段(纯常规行人)  → D | m(路人/其他正常活动)
    Test  片段(含骑车/滑板/小车闯入) → C | s(可疑徘徊,粗粒度占位)
  ⚠️ 正式代理训练前须用 annotation/gemini_labeler.py 重打伪 GT
    (--mode double 双温度一致性过滤),本脚本产出的 whitelist 为空。

用法:
  python data/prepare_proxy_ucsd.py --src /tmp/UCSD_Anomaly_Dataset.v1p2 \
      --subset ped2 --out /tmp/proxy_ucsd
产出:
  out/videos/*.mp4(tif 帧序列 @10fps 合成)
  out/DATA/labels.jsonl(video_id/uri/duration/resolution/labels/meta)
  out/DATA/asset_*.{jsonl,txt}(空占位,结构齐全)
"""
import argparse
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FPS = 10.0        # UCSD 官方采集帧率

DESC = {
    "train": "A pedestrian walks along the outdoor walkway past the "
             "camera during the daytime.",
    "test": "A person moves through the walkway in an unusual manner, "
            "lingering or crossing against the pedestrian flow.",
}


def clip_dirs(src, subset):
    root = os.path.join(src, subset.replace("ped", "UCSDped"))
    for split in ("Train", "Test"):
        for d in sorted(glob.glob(os.path.join(root, split, f"{split}[0-9]*"))):
            if os.path.isdir(d) and not d.endswith("_gt"):
                yield split.lower(), d


def frames_to_mp4(frame_dir, out_path):
    tifs = sorted(glob.glob(os.path.join(frame_dir, "*.tif")))
    if not tifs:
        return None
    first = cv2.imread(tifs[0])
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                         FPS, (w, h))
    for t in tifs:
        vw.write(cv2.imread(t))
    vw.release()
    return len(tifs), w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="解包后的 UCSD 根目录")
    ap.add_argument("--subset", default="ped2", choices=["ped1", "ped2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="最多处理条数")
    a = ap.parse_args()

    vdir = os.path.join(a.out, "videos")
    ddir = os.path.join(a.out, "DATA")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(ddir, exist_ok=True)

    records, n = [], 0
    for split, d in clip_dirs(a.src, a.subset):
        if a.limit and n >= a.limit:
            break
        vid = f"{a.subset}_{split}_{os.path.basename(d).lower()}"
        out_mp4 = os.path.join(vdir, f"{vid}.mp4")
        r = frames_to_mp4(d, out_mp4)
        if r is None:
            continue
        n_frames, w, h = r
        rt, sk = ("D", "m") if split == "train" else ("C", "s")
        records.append({
            "video_id": vid,
            "video_uri": f"{vid}.mp4",
            "duration_sec": round(n_frames / FPS, 2),
            "resolution": f"{w}x{h}",
            "labels": {"role_type": rt, "sub_keyscene": sk,
                       "description": DESC[split]},
            "meta": {"camera_id": f"ucsd_{a.subset}",
                     "label_source": "rule_pseudo",   # 待 Gemini 重打
                     "source_split": split},
        })
        n += 1
        print(f"  {vid}: {n_frames}f {w}x{h} -> {rt}|{sk}")

    with open(os.path.join(ddir, "labels.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n"
                     for r in records)
    # 资产占位(结构齐全,内容待 gemini_labeler 产出)
    open(os.path.join(ddir, "asset_A_attributes.jsonl"), "w").close()
    open(os.path.join(ddir, "asset_C_reasoning.jsonl"), "w").close()
    open(os.path.join(ddir, "asset_D_whitelist.txt"), "w").close()
    print(f"\n{len(records)} clips -> {a.out}(labels 为规则伪标,"
          f"whitelist 为空 —— 正式训练前跑 gemini_labeler)")


if __name__ == "__main__":
    main()
