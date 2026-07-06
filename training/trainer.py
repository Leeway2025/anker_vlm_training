"""加权 SFT Trainer(TPU/torch_xla)。

损失 = token 加权 CE(label_smoothing 0.1,分类 token ×4,think 0)
     + 0.2 × KS 父类头(Phase 5)
     + 0.3 × 7 属性辅助头(Phase 5b)

TPU 注意:
  - HF Trainer 经 accelerate 原生支持 XLA;启动用
    `python xla_spawn.py --num_cores 8 training/train.py …` 或 torchrun+xla
  - collator 已固定 padding(base.yaml max_seq_len),避免 XLA 重编译
  - 自定义键(token_weights/ks_labels/aux_labels)在 compute_loss 中 pop
"""
import torch
import torch.nn.functional as F
from transformers import Trainer


class WeightedSFTTrainer(Trainer):
    def __init__(self, *args, run_cfg=None, ks_head=None, aux_heads=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.run_cfg = run_cfg
        self.ks_head = ks_head
        self.aux_heads = aux_heads
        self._loss_log = {"cls": 0.0, "desc": 0.0, "n": 0}

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        weights = inputs.pop("token_weights")
        ks_labels = inputs.pop("ks_labels", None)
        aux_labels = inputs.pop("aux_labels", None)
        labels = inputs["labels"]

        need_hidden = self.ks_head is not None
        outputs = model(**inputs, output_hidden_states=need_hidden)
        logits = outputs.logits

        # shift
        s_logits = logits[:, :-1].float()
        s_labels = labels[:, 1:]
        s_weights = weights[:, 1:]

        ls = self.run_cfg["loss"]["label_smoothing"]
        ce = F.cross_entropy(
            s_logits.reshape(-1, s_logits.size(-1)),
            s_labels.reshape(-1),
            reduction="none", ignore_index=-100, label_smoothing=ls,
        ).view(s_labels.shape)

        valid = (s_labels != -100).float()
        w = s_weights * valid
        loss = (ce * w).sum() / w.sum().clamp_min(1.0)

        # 监控: 分类 loss 与 desc loss 分开(训练监控清单要求 —— 验证
        # "分类 loss 在降",不能只有 description 在降)
        with torch.no_grad():
            cls_m = (s_weights > 1.0) & (s_labels != -100)
            desc_m = (s_weights == 1.0) & (s_labels != -100)
            if cls_m.any():
                self._loss_log["cls"] += ce[cls_m].mean().item()
            if desc_m.any():
                self._loss_log["desc"] += ce[desc_m].mean().item()
            self._loss_log["n"] += 1

        if self.ks_head is not None and ks_labels is not None:
            hidden = outputs.hidden_states[-1]
            mask = (labels != -100)
            loss = loss + self.run_cfg["loss"]["ks_parent_coef"] * \
                self.ks_head.compute_loss(hidden, mask, ks_labels)

        if self.aux_heads is not None and aux_labels is not None:
            loss = loss + self.run_cfg["loss"]["aux_coef"] * \
                self.aux_heads.compute_loss(aux_labels)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        n = max(self._loss_log["n"], 1)
        logs["loss_cls"] = round(self._loss_log["cls"] / n, 4)
        logs["loss_desc"] = round(self._loss_log["desc"] / n, 4)
        self._loss_log = {"cls": 0.0, "desc": 0.0, "n": 0}
        return super().log(logs, *args, **kwargs)
