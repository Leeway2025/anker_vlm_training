"""JAX 路线数据管线(独立实现,零 torch 依赖)。

样本 = 模板 ids(HF 基准排布,Gate B 已逐位对齐)+ label ids + 权重:
  - 模板来自 poc/02a 导出的 hf_layout.json(生产 prompt + 16 帧排布),
    视频占位 258884 → 哨兵 -2(Gate C 配方)
  - label: "{RT} | {SubKS} | {desc}" 编码;分类段(第二个 | 之前)
    token 权重 ×4,desc 段 ×1(与 torch 侧 loss 设计一致)
  - 固定 padding 到 max_len(XLA 静态形状)
帧: euno-wds 分片直读(tar 内 {video_id}.pyd pickle,16×JPEG bytes)。
"""
import io
import json
import os
import pickle
import tarfile

import numpy as np

EOT = 106            # <end_of_turn>(Gemma4 对话轮结束,作 label 终止符)
SENTINEL = -2        # 视觉哨兵(gemma4 JAX 模型的 TOKEN_PLACEHOLDER)


def load_frames(rec, wds_dir):
    meta = rec.get("meta") or {}
    shard = os.path.join(wds_dir, f"shard-{meta.get('shard', 0):06d}.tar")
    with tarfile.open(shard) as tf:
        raw = tf.extractfile(f"{rec['video_id']}.pyd").read()
    frames = pickle.loads(raw)["frames"]
    from PIL import Image
    return [np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
            for b in frames]


class SftDataset:
    def __init__(self, labels_file, layout_file, tokenizer, wds_dir=None,
                 max_label_len=64, cls_weight=4.0):
        self.recs = [json.loads(l) for l in open(labels_file, encoding="utf-8")]
        self.wds = wds_dir or os.path.dirname(labels_file)
        lay = json.load(open(layout_file, encoding="utf-8"))
        ids = lay["input_ids"]
        mm = lay["mm_token_type_ids"]
        self.template = [(SENTINEL if m == 2 else t) for t, m in zip(ids, mm)]
        self.tok = tokenizer
        self.max_label_len = max_label_len
        self.cls_w = cls_weight
        self.max_len = len(self.template) + max_label_len

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        lb = rec["labels"]
        # 分类段 ×cls_w,desc 段 ×1(与 torch collator 的 token 加权同构)
        cls_ids = self.tok.encode(
            f"{lb['role_type']} | {lb['sub_keyscene']} |")
        desc_ids = self.tok.encode(f" {lb['description']}") + [EOT]
        lab = (list(cls_ids) + list(desc_ids))[: self.max_label_len]
        w = ([self.cls_w] * len(cls_ids) + [1.0] * len(desc_ids))[
            : self.max_label_len]

        L, T = self.max_len, len(self.template)
        tokens = np.zeros(L, np.int32)
        labels = np.full(L, -100, np.int32)
        weights = np.zeros(L, np.float32)
        tokens[:T] = self.template
        tokens[T:T + len(lab)] = lab
        labels[T:T + len(lab)] = lab
        weights[T:T + len(lab)] = w

        frames = load_frames(rec, self.wds)
        return {"tokens": tokens, "labels": labels, "weights": weights,
                "frames": frames, "video_id": rec["video_id"]}


def make_vision_input(frames_list):
    """B 个样本(各 16 帧)→ 模型入参。v1 限 bs=1(见 FINDINGS)。"""
    from gemma.gm.nn.gemma4.vision._preprocessing import preprocess_and_patchify
    assert len(frames_list) == 1, "v1 每设备 bs=1;多样本拼接语义待确认"
    patches, pos, counts = preprocess_and_patchify(
        frames_list[0], max_soft_tokens=64)
    n, p, d = patches.shape
    return (patches.reshape(1, n * p, d), pos.reshape(1, n * p, 2),
            tuple(int(c) for c in counts))
