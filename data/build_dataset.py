"""训练数据集与 collator。

样本流:
  labels.jsonl(+资产 A 属性 / 资产 C 推理链 / 资产 D 白名单)
    → AnkerVideoDataset.__getitem__: 解码+增强+resize(numpy)
    → AnkerCollator: processor 编码,labels 掩 prompt(-100),
      token_weights 由字符 span 映射(分类 ×4 / think 0 / desc 1)

需 GPU 环境烟测的点(标 SMOKE):processor 的 chat template 与
video 输入姿势因 transformers 版本而异,collator 内做了两级回退。
"""
import json
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.formatting import (build_target, build_cot_target,        # noqa: E402
                             char_spans_to_token_weights, format_reasoning)
from data.taxonomy import (aux_label_index, AUX_HEAD_ORDER,          # noqa: E402
                           KS_GROUP, KS_CLASSES,
                           view_type_from_resolution)


def load_jsonl(path, key="video_id"):
    out = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            out[d[key]] = d
    return out


def split_by_camera(records, val_size, holdout_key="camera_id", seed=0):
    """按摄像头/住户切分,防"记住这个门廊"式泄漏(training_plan 2.1)。
    无 camera_id 的样本退化为按 video_id 切(记录 warning)。"""
    rng = random.Random(seed)
    by_cam = {}
    for r in records:
        cam = (r.get("meta") or {}).get(holdout_key) or r["video_id"]
        by_cam.setdefault(cam, []).append(r)
    cams = sorted(by_cam)
    rng.shuffle(cams)
    val, val_n = [], 0
    for c in cams:
        if val_n >= val_size:
            break
        val += by_cam[c]
        val_n += len(by_cam[c])
    val_ids = {r["video_id"] for r in val}
    train = [r for r in records if r["video_id"] not in val_ids]
    return train, val


class AnkerVideoDataset:
    """torch.utils.data.Dataset 兼容(不强依赖 torch,便于逻辑测试)。"""

    def __init__(self, records, cfg, phase_cfg, training=True,
                 attributes=None, reasoning=None, sample_weights=None):
        self.records = records
        self.cfg = cfg
        self.phase = phase_cfg
        self.training = training
        self.attributes = attributes or {}
        self.reasoning = reasoning or {}
        self.sample_weights = sample_weights or {}
        self.rng = random.Random(cfg.get("seed", 0))
        self._anneal = False          # Phase 5d 退火期: 纯生产模式

    def set_anneal(self, on: bool):
        self._anneal = on

    def __len__(self):
        return len(self.records)

    # ---------- 纯逻辑部分(可单测): 构造文本与标签 ----------
    def build_text_example(self, rec):
        lab = rec.get("labels", rec)
        rt, sk = lab["role_type"], lab["sub_keyscene"]
        desc = lab.get("description", "")
        fmt = self.cfg["format"]
        sep = fmt["separator"]
        clsw = self.cfg["loss"]["cls_token_weight"]

        cot = bool(self.phase.get("cot_mode")) and not self._anneal
        use_reason = False
        if cot and rec["video_id"] in self.reasoning:
            use_reason = self.rng.random() < self.phase.get("cot_reason_ratio", 0.6)

        if use_reason:
            r = self.reasoning[rec["video_id"]]
            reasoning = r.get("reasoning_chain") or format_reasoning(
                r.get("identity_clues", ""), r.get("scene_clues", ""),
                r.get("conclusion", ""))
            spec = build_cot_target(rt, sk, desc, reasoning, sep, clsw,
                                    fmt["think_open"], fmt["think_close"])
            prompt_suffix = "\n" + fmt["reason_marker"]
        else:
            spec = build_target(rt, sk, desc, sep, clsw)
            prompt_suffix = ""

        # 辅助头标签(Phase 5b): 资产 A + view_type 分辨率规则覆盖
        aux = {}
        if self.phase.get("enable_aux_heads"):
            attrs = (self.attributes.get(rec["video_id"], {})
                     .get("attributes", self.attributes.get(rec["video_id"], {})))
            conf = self.attributes.get(rec["video_id"], {}).get("confidence", 1.0)
            low_conf = conf < self.phase.get("aux_conf_threshold", 0.5)
            for head in AUX_HEAD_ORDER:
                v = attrs.get(head)
                if head == "view_type":       # 分辨率规则优先(免费且更准)
                    res = rec.get("resolution") or ""
                    if "x" in str(res):
                        w, h = (int(x) for x in str(res).split("x"))
                        rule = view_type_from_resolution(w, h)
                        v = rule or v
                aux[head] = -100 if (v is None or low_conf) \
                    else aux_label_index(head, v)

        ks_label = KS_CLASSES.index(KS_GROUP[sk]) \
            if self.phase.get("enable_ks_parent_head") else -100

        return {"target_spec": spec, "prompt_suffix": prompt_suffix,
                "aux_labels": aux, "ks_label": ks_label,
                "rt": rt, "sk": sk}

    def __getitem__(self, i):
        rec = self.records[i]
        ex = self.build_text_example(rec)

        from data.sampling import decode_video, resize_production, uniform_indices
        from data.augmentation import plan_augmented_indices, spatial_augment
        path = os.path.join(self.cfg["data"]["video_root"],
                            os.path.basename(rec.get("video_uri", rec["video_id"])))
        num_frames = self.cfg["sampling"]["num_frames"]

        if self.training:
            # 需要总帧数来做时序裁剪 → decode 分两步
            import cv2
            cap = cv2.VideoCapture(path)
            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or num_frames
            cap.release()
            idx = plan_augmented_indices(n_total, num_frames,
                                         self.cfg["augment"], self.rng)
            frames, _ = decode_video(path, indices=idx)
            frames = spatial_augment(frames, self.cfg["augment"], self.rng)
        else:
            frames, _ = decode_video(path, num_frames=num_frames)

        frames = resize_production(frames, self.cfg["sampling"]["image_size"])
        return {"frames": frames, **ex, "video_id": rec["video_id"]}


class AnkerCollator:
    """processor 编码 + labels/token_weights 构造。

    Gemma4 processor 真实签名(2026-07-07 v6e-1 真机烟测确认):
      - 视频经 chat template `{"type": "video", "video": frames}` 传入,
        必须 `do_sample_frames=False`(内置采样器默认重采 32 帧,
        与"均匀 16 帧照抄生产"冲突)
      - 输出 pixel_values_videos (T,630,768) / video_position_ids /
        mm_token_type_ids;70 token/帧 = 630 patch 经 3×3 pooling
      - 帧时间戳写进 prompt,未传 video_metadata 时按 fps=24 兜底
        (TODO 客户对齐: 生产端时间戳/metadata 约定,训练须照抄)
    """

    MM_KEYS = ("pixel_values_videos", "video_position_ids")

    def __init__(self, processor, cfg):
        self.p = processor
        self.cfg = cfg
        self.prompt_text = open(cfg["format"]["prompt_file"],
                                encoding="utf-8").read().strip()

    def _encode_prompt(self, frames, suffix):
        """<video>+生产 prompt(+可选 [REASON])→ processor 编码结果
        (prompt 段 ids 含视觉 token 与时间戳,及视频张量)。"""
        text = self.prompt_text + suffix
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": text}]}]
        return self.p.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", do_sample_frames=False)

    def __call__(self, batch):
        import torch
        input_ids_l, labels_l, weights_l, mmtt_l = [], [], [], []
        pixel_l, vpos_l, aux_l, ks_l = [], [], [], []
        tok = self.p.tokenizer
        model_dtype = getattr(torch, self.cfg["model"]["torch_dtype"])

        for ex in batch:
            enc = self._encode_prompt(ex["frames"], ex["prompt_suffix"])
            p_ids = enc["input_ids"][0].tolist()
            mm_tt = enc["mm_token_type_ids"][0].tolist()

            spec = ex["target_spec"]
            t_enc = tok(spec.text, add_special_tokens=False,
                        return_offsets_mapping=True)
            t_ids = t_enc["input_ids"] + [tok.eos_token_id]
            t_w = char_spans_to_token_weights(spec, t_enc["offset_mapping"])
            t_w = t_w + [1.0]                                 # eos 权重 1

            input_ids_l.append(torch.tensor(p_ids + t_ids))
            labels_l.append(torch.tensor([-100] * len(p_ids) + t_ids))
            weights_l.append(torch.tensor([0.0] * len(p_ids) + t_w))
            mmtt_l.append(torch.tensor(mm_tt + [0] * len(t_ids)))
            pixel_l.append(enc["pixel_values_videos"][0].to(model_dtype))
            vpos_l.append(enc["video_position_ids"][0])
            ks_l.append(ex["ks_label"])
            aux_l.append([ex["aux_labels"].get(h, -100)
                          for h in AUX_HEAD_ORDER])

        pad = tok.pad_token_id or 0
        from torch.nn.utils.rnn import pad_sequence

        def _pad(seqs, value):
            x = pad_sequence(seqs, batch_first=True, padding_value=value)
            # TPU/XLA: 固定长度 padding,否则每个 batch 触发一次重编译
            if self.cfg["train"].get("pad_to_fixed_length"):
                L = self.cfg["train"]["max_seq_len"]
                if x.shape[1] > L:
                    x = x[:, :L]      # 超长截断(监控 truncation 率)
                elif x.shape[1] < L:
                    padder = x.new_full((x.shape[0], L - x.shape[1]), value)
                    x = torch.cat([x, padder], dim=1)
            return x

        out = {
            "input_ids": _pad(input_ids_l, pad),
            "labels": _pad(labels_l, -100),
            "token_weights": _pad(weights_l, 0.0),
            "mm_token_type_ids": _pad(mmtt_l, 0),
            "pixel_values_videos": torch.stack(pixel_l),
            "video_position_ids": torch.stack(vpos_l),
            "ks_labels": torch.tensor(ks_l),
            "aux_labels": torch.tensor(aux_l),
        }
        out["attention_mask"] = (out["input_ids"] != pad).long()
        return out
