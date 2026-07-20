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
    """分片定位优先级: meta.wds_dir(labels.jsonl 内声明,与 torch 侧
    euno_wds 行为一致)> 调用方传入的 wds_dir(--wds-dir / labels 同目录)。
    容器场景注意: meta.wds_dir 必须写容器内可见的路径 —— 最省事的做法是
    把分片目录以同名路径挂载(-v /真实路径:/真实路径),jsonl 零修改。"""
    meta = rec.get("meta") or {}
    base = meta.get("wds_dir") or wds_dir
    shard = os.path.join(base, f"shard-{meta.get('shard', 0):06d}.tar")
    with tarfile.open(shard) as tf:
        # 成员名约定与 torch 侧 euno_wds 一致: video_id 中的 "/" → "__"
        name = rec["video_id"].replace("/", "__") + ".pyd"
        try:
            raw = tf.extractfile(tf.getmember(name)).read()
        except KeyError:
            few = [m.name for m in tf.getmembers()[:3]]
            raise KeyError(f"{shard} 中无成员 {name!r};分片内实际成员形如 "
                           f"{few} —— 若命名约定不同请反馈")
    frames = pickle.loads(raw)["frames"]
    from PIL import Image
    return [np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
            for b in frames]


def load_jsonl_map(path):
    return {j["video_id"]: j for j in
            (json.loads(l) for l in open(path, encoding="utf-8"))}


def split_by_camera(recs, val_size, seed=0):
    """与 torch 侧 build_dataset.split_by_camera 同语义: 按摄像头整组
    进 val,防"记住这个门廊"式泄漏;无 camera_id / unknown 退化为按
    video_id。固定 seed → val 集跨运行稳定。"""
    rng = random.Random(seed)
    by_cam = {}
    for r in recs:
        cam = (r.get("meta") or {}).get("camera_id") or r["video_id"]
        if cam == "unknown":
            cam = r["video_id"]
        by_cam.setdefault(cam, []).append(r)
    cams = sorted(by_cam)
    rng.shuffle(cams)
    val, n = [], 0
    for c in cams:
        if n >= val_size:
            break
        val += by_cam[c]
        n += len(by_cam[c])
    val_ids = {r["video_id"] for r in val}
    return [r for r in recs if r["video_id"] not in val_ids], val


class SftDataset:
    def __init__(self, labels_file, layout_file, tokenizer, wds_dir=None,   # wds_dir 显式传入时覆盖 meta(见 load_frames)
                 max_label_len=64, cls_weight=4.0, sample_weights=None,
                 reasoning=None, cot_ratio=0.6, attributes=None,
                 max_think_len=96, seed=0, val_n=0):
        recs = [json.loads(l) for l in open(labels_file, encoding="utf-8")]
        # 顺序铁律: 先切 val、再对 train 做 hard-mining 复制 —— 反过来
        # 副本会横跨 train/val(泄漏,val loss 虚低)。torch 侧同序。
        if val_n:
            train_recs, val_recs = split_by_camera(recs, val_n, seed=seed)
        else:
            train_recs, val_recs = recs, []
        if sample_weights:      # hard-mining 物理复制(流式最大余数法:
            out = []            # round 会把 1.0<w<1.5 全截成 1,类上限失效)
            acc = 0.0
            for r in train_recs:
                w = max(1.0, float(sample_weights.get(r["video_id"], 1.0)))
                n = int(w)
                acc += w - n
                if acc >= 1.0:
                    n += 1
                    acc -= 1.0
                out.extend([r] * n)
            print(f"[hard-mining] train {len(train_recs)} -> {len(out)}")
            train_recs = out
        self.recs = train_recs + val_recs
        self.first_val = len(train_recs)        # ≥此下标 = val(无 CoT 注入)
        self.train_idx = list(range(len(train_recs)))
        self.val_idx = list(range(len(train_recs), len(self.recs)))
        self.wds_override = wds_dir            # 显式指定则最高优先
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
        # val 样本(i >= first_val)永不注入 CoT: 保证 val loss 分布固定、
        # 各次 eval 可比 —— 否则 best checkpoint 选择近似掷骰子
        if (i < self.first_val and self.reasoning.get(vid)
                and not self.anneal
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

        # 优先级: 显式 wds_dir(--wds-dir)> meta.wds_dir > labels 同目录
        if self.wds_override:
            frames = load_frames({**rec, "meta": {**(rec.get("meta") or {}),
                                                  "wds_dir": self.wds_override}},
                                 self.wds)
        else:
            frames = load_frames(rec, self.wds)
        return {"tokens": tokens, "labels": labels, "weights": weights,
                "aux_labels": aux, "ks_label": np.int32(ks),
                "frames": frames, "video_id": vid}


def make_vision_input(frames_list):
    """B 个样本(各 16 帧)→ 模型入参 [B, n*p, ·]。
    B>1 需先 install_batched_encode_vision()(poc/05 等价测试 PASS)。"""
    from gemma.gm.nn.gemma4.vision._preprocessing import preprocess_and_patchify
    pas, poss, counts0 = [], [], None
    for frames in frames_list:
        patches, pos, counts = preprocess_and_patchify(
            frames, max_soft_tokens=64)
        n, p, d = patches.shape
        pas.append(patches.reshape(n * p, d))
        poss.append(pos.reshape(n * p, 2))
        counts = tuple(int(c) for c in counts)
        assert counts0 in (None, counts), "batch 内 counts 必须一致"
        counts0 = counts
    return (np.stack(pas), np.stack(poss), counts0)


def install_batched_encode_vision():
    """gm 官方 _encode_vision 写死 B=1(reshape 忽略 batch 维);merge 侧
    vmap 天然支持 [B,T,D]。此补丁 B=1 走原路径,B>1 批量展开编码后折回。
    语义经 poc/05 等价测试钉死: batch=2 ≡ 2×bs1,max|Δ|<1e-4。"""
    import jax.numpy as jnp
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    if getattr(g4_tr, "_BATCH_EV_PATCHED", False):
        return
    _orig = g4_tr.Transformer._encode_vision

    def _batched(self, vision_input):
        patches = vision_input.patches
        B = patches.shape[0]
        if B == 1:
            return _orig(self, vision_input)
        counts = vision_input.soft_token_counts
        n, cnt = len(counts), counts[0]
        assert all(c == cnt for c in counts), "非均匀 counts 需回退 bs=1"
        p, d = patches.shape[1] // n, patches.shape[2]
        pa = jnp.reshape(patches, (B * n, p, d))
        px = jnp.reshape(vision_input.positions_xy, (B * n, p, 2))
        emb, _mask = self.vision_encoder(pa, px)[0]
        toks = emb[:, :cnt, :]                # 均匀正方形帧无 pad → 前 cnt 即真
        toks = jnp.reshape(toks, (B, n * cnt, toks.shape[-1]))
        return self.embedder.encode_vision(toks[:, None])[:, 0]

    g4_tr.Transformer._encode_vision = _batched
    g4_tr._BATCH_EV_PATCHED = True
