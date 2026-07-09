"""Euno WebDataset(客户真实数据格式)读取路径。

依据 euno数据集说明.md(2026-07-08,客户提供):
  - 数据: shard-*.tar 内 .pyd(pickle: frames=16×JPEG bytes 384×384 RGB,
    video_rel, num_frames);index.json: sample_key → shard_id;
    tar 内文件名 = sample_key 的 "/" 替换为 "__"
  - 标注: LlamaFactory 对话 json,gpt value = "RT|SK|desc"(无空格,已核对)
  - ⚠️ 上游已完成均匀 16 帧采样与 384×384 resize → 本路径:
    ① 不再做时序裁剪/原图 RandomCrop(拿不到原片,增强空间受限,
       仅保留水平翻转/亮度对比度/帧 dropout)
    ② 无 resolution 字段 → view_type 分辨率规则不可用,只能靠 Gemini
    ③ 原始时长丢失 → duration_sec 置 null
  - camera_id: video_rel 文件名内的设备序列号(T8 开头 token)>
    无序列号(uuid 命名)记 "unknown",可再用 camera_fingerprint 兜底

用法(转换标注 → 通用 labels.jsonl):
  python -m data.euno_wds --annotation euno_train_xxx.json \
      --wds-dir anker_video_clips_wds_full --out DATA/labels.jsonl
训练时 base.yaml data.labels_file 指向产出文件,
train.py 检测 meta.storage=="wds" 自动走本数据集类。
"""
import argparse
import io
import json
import os
import pickle
import re
import sys
import tarfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.build_dataset import AnkerVideoDataset          # noqa: E402

_DEVICE_SN = re.compile(r"(T8[0-9A-Za-z]{6,})")


def camera_from_video_rel(video_rel: str) -> str:
    """文件名内设备序列号 → camera_id;uuid 命名 → 'unknown'
    (可后接 data/camera_fingerprint.py 兜底)。"""
    m = _DEVICE_SN.search(os.path.basename(video_rel))
    return m.group(1) if m else "unknown"


def parse_gpt_label(value: str):
    """'RT|SK|desc'(无空格,euno 核对版式)→ (rt, sk, desc)。"""
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"bad label: {value!r}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def euno_to_labels(annotation_path: str, wds_dir: str, out_path: str,
                   limit: int = 0):
    """LlamaFactory 标注 json → 通用 labels.jsonl(meta.storage='wds')。"""
    anns = json.load(open(annotation_path, encoding="utf-8"))
    index = json.load(open(os.path.join(wds_dir, "index.json"),
                           encoding="utf-8"))
    n_bad = n_noshard = 0
    out = []
    for ann in anns:
        if limit and len(out) >= limit:
            break
        key = ann["videos"][0]
        try:
            rt, sk, desc = parse_gpt_label(ann["conversations"][1]["value"])
        except (ValueError, IndexError, KeyError):
            n_bad += 1
            continue
        shard = index.get(key)
        if shard is None:
            n_noshard += 1
            continue
        out.append({
            "video_id": key,
            "video_uri": key,
            "duration_sec": None,          # 上游采样后原时长丢失
            "resolution": "384x384",       # 上游已 resize,原分辨率丢失
            "labels": {"role_type": rt, "sub_keyscene": sk,
                       "description": desc},
            "meta": {"camera_id": camera_from_video_rel(key),
                     "storage": "wds", "wds_dir": wds_dir,
                     "shard": shard, "euno_id": ann.get("id")},
        })
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    cams = len({r["meta"]["camera_id"] for r in out})
    n_unknown = sum(1 for r in out if r["meta"]["camera_id"] == "unknown")
    print(f"{len(out)} samples -> {out_path} | cameras={cams} "
          f"(unknown={n_unknown}, 建议对 unknown 跑 camera_fingerprint) | "
          f"跳过: 标签坏={n_bad}, 无分片={n_noshard}")
    return out


class EunoWDSDataset(AnkerVideoDataset):
    """帧来源 = WDS tar 分片(其余逻辑复用 AnkerVideoDataset)。

    增强限制(上游已采样/resize): 只做水平翻转 + 亮度对比度 + 帧 dropout;
    时序裁剪与空间 crop 被强制关闭。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tar_cache = {}

    def _load_pyd(self, rec):
        wds_dir, shard = rec["meta"]["wds_dir"], rec["meta"]["shard"]
        path = os.path.join(wds_dir, f"shard-{shard:06d}.tar")
        tf = self._tar_cache.get(path)
        if tf is None:
            tf = tarfile.open(path)
            self._tar_cache[path] = tf
        name = rec["video_id"].replace("/", "__") + ".pyd"
        return pickle.load(tf.extractfile(tf.getmember(name)))

    def __getitem__(self, i):
        import cv2
        rec = self.records[i]
        ex = self.build_text_example(rec)
        data = self._load_pyd(rec)
        frames = np.stack([
            cv2.cvtColor(cv2.imdecode(
                np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB)
            for b in data["frames"]])
        if self.training:
            from data.augmentation import spatial_augment
            aug = dict(self.cfg["augment"])
            aug["spatial_crop_scale"] = (1.0, 1.0)   # 原图不可得,禁 crop
            frames = spatial_augment(frames, aug, self.rng)
            p = self.cfg["augment"].get("frame_dropout_prob", 0.0)
            for t in range(1, len(frames)):          # 帧 dropout(数组级)
                if self.rng.random() < p:
                    frames[t] = frames[t - 1]
        return {"frames": frames, **ex, "video_id": rec["video_id"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", required=True,
                    help="euno LlamaFactory 标注 json")
    ap.add_argument("--wds-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    euno_to_labels(a.annotation, a.wds_dir, a.out, a.limit)


if __name__ == "__main__":
    main()
