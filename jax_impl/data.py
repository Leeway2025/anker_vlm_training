"""JAX 路线数据管线(独立实现,零 torch 依赖)。

样本 = 模板 ids(HF 基准排布,Gate B 已逐位对齐)+ label ids + 权重:
  - 模板来自 poc/02a 导出的 hf_layout.json(生产 prompt + 16 帧排布),
    视频占位 258884 → 哨兵 -2(Gate C 配方)
  - label: "{RT} | {SubKS} | {desc}";分类段 ×4,desc ×1,think 段 ×0
    (与 torch 侧 loss 设计一致)
  - 固定 padding 到 max_len(XLA 静态形状)
支持: hard-mining 物理复制 / implicit-CoT(比例混合+退火)/ aux 标签。
帧: euno-wds 分片直读(tar 内 {video_id}.pyd pickle,16×JPEG bytes)。
"""
import io
import json
import os
import pickle
import random
import sys
import tarfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.taxonomy import AUX_HEAD_ORDER, aux_label_index, KS_GROUP, KS_CLASSES  # noqa: E402

EOT = 106            # <end_of_turn>
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


def load_jsonl_map(path):
    return {j["video_id"]: j for j in
            (json.loads(l) for l in open(path, encoding="utf-8"))}


class SftDataset:
    def __init__(self, labels_file, layout_file, tokenizer, wds_dir=None,
                 max_label_len=64, cls_weight=4.0, sample_weights=None,
                 reasoning=None, cot_ratio=0.6, attributes=None,
                 max_think_len=96, seed=0):
        recs = [json.loads(l) for l in open(labels_file, encoding="utf-8")]
        if sample_weights:                      # hard-mining 物理复制
            out = []
            for r in recs:
                n = max(1, round(sample_weights.get(r["video_id"], 1.0)))
                out.extend([r] * n)
            print(f"[hard-mining] {len(recs)} -> {len(out)} samples")
            recs = out
        self.recs = recs
        self.wds = wds_dir or os.path.dirname(labels_file)
        lay = json.load(open(layout_file, encoding="utf-8"))
        self.template = [(SENTINEL if m == 2 else t) for t, m in
                         zip(lay["input_ids"], lay["mm_token_type_ids"])]
        self.tok = tokenizer
        self.max_label_len = max_label_len
        self.cls_w = cls_weight
        self.reasoning = reasoning or {}        # video_id → 资产 C
        self.cot_ratio = cot_ratio
        self.anneal = False                     # True → 纯生产模式
        self.max_think = max_think_len if self.reasoning else 0
        self.attributes = attributes or {}      # video_id → 资产 A
        self.rng = random.Random(seed)
        self.max_len = len(self.template) + self.max_think + max_label_len

    def set_anneal(self, flag):                 # CoT 退火(torch 同款语义)
        self.anneal = flag

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        lb = rec["labels"]
        vid = rec["video_id"]
        cls_ids = self.tok.encode(
            f"{lb['role_type']} | {lb['sub_keyscene']} |")
        desc_ids = self.tok.encode(f" {lb['description']}") + [EOT]

        think_ids = []
        if (self.reasoning.get(vid) and not self.anneal
                and self.rng.random() < self.cot_ratio):
            r = self.reasoning[vid]
            txt = r.get("reasoning_chain") or (
                f"[Identity cues] {r.get('identity_clues', '')} "
                f"[Scene cues] {r.get('scene_clues', '')} "
                f"[Conclusion] {r.get('conclusion', '')}")
            think_ids = self.tok.encode(txt)[: self.max_think]

        lab = (list(think_ids) + list(cls_ids) + list(desc_ids))
        w = ([0.0] * len(think_ids)             # think 段权重 0(隐式 CoT)
             + [self.cls_w] * len(cls_ids) + [1.0] * len(desc_ids))
        cap = self.max_think + self.max_label_len
        lab, w = lab[:cap], w[:cap]

        L, T = self.max_len, len(self.template)
        tokens = np.zeros(L, np.int32)
        labels = np.full(L, -100, np.int32)
        weights = np.zeros(L, np.float32)
        tokens[:T] = self.template
        tokens[T:T + len(lab)] = lab
        labels[T:T + len(lab)] = lab
        weights[T:T + len(lab)] = w

        # 辅助头标签: 资产 A 的 7 属性 + KS 父类(6 类)
        attrs = self.attributes.get(vid, {})
        av = attrs.get("attributes", attrs)
        aux = np.array([aux_label_index(h, av.get(h, "")) if av else -100
                        for h in AUX_HEAD_ORDER], np.int32)
        ks = KS_CLASSES.index(KS_GROUP[lb["sub_keyscene"]]) \
            if lb["sub_keyscene"] in KS_GROUP else -100

        frames = load_frames(rec, self.wds)
        return {"tokens": tokens, "labels": labels, "weights": weights,
                "aux_labels": aux, "ks_label": np.int32(ks),
                "frames": frames, "video_id": vid}


def make_vision_input(frames_list):
    """B 个样本(各 16 帧)→ 模型入参。v1 限 bs=1(见 FINDINGS)。"""
    from gemma.gm.nn.gemma4.vision._preprocessing import preprocess_and_patchify
    assert len(frames_list) == 1, "v1 每设备 bs=1;多样本拼接语义待确认"
    patches, pos, counts = preprocess_and_patchify(
        frames_list[0], max_soft_tokens=64)
    n, p, d = patches.shape
    return (patches.reshape(1, n * p, d), pos.reshape(1, n * p, 2),
            tuple(int(c) for c in counts))
