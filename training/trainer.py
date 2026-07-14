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
        # 必须 pop: labels 若留在 inputs 进 forward,Gemma4 会额外计算内置
        # 全词表 CE,物化 (B,L-1,262k) logits 及中间量(bs2 实测 +16GB,
        # 直接 OOM;transformers 5.13 不会 DCE 这条未使用的 loss 路径)
        labels = inputs.pop("labels")

        need_hidden = self.ks_head is not None
        outputs = model(**inputs, output_hidden_states=need_hidden)
        logits = outputs.logits

        # shift(保持 bf16,分块内再升 fp32 —— 整段 .float() 会物化
        # (B,L,V) fp32: bs8×L2047×V262k ≈ 17GB,v6e 单芯必 OOM,真机确认)
        s_logits = logits[:, :-1]
        s_labels = labels[:, 1:]
        s_weights = weights[:, 1:]

        ls = self.run_cfg["loss"]["label_smoothing"]
        V = s_logits.size(-1)
        chunk = int(self.run_cfg["train"].get("ce_chunk", 256))
        loss_num = logits.new_zeros((), dtype=torch.float32)
        loss_den = logits.new_zeros((), dtype=torch.float32)
        cls_sum = logits.new_zeros((), dtype=torch.float32)
        cls_cnt = logits.new_zeros((), dtype=torch.float32)
        desc_sum = logits.new_zeros((), dtype=torch.float32)
        desc_cnt = logits.new_zeros((), dtype=torch.float32)
        for s in range(0, s_labels.shape[1], chunk):     # 步数静态,XLA 单编译
            lg = s_logits[:, s:s + chunk].float()
            lb = s_labels[:, s:s + chunk]
            wt = s_weights[:, s:s + chunk]
            ce = F.cross_entropy(
                lg.reshape(-1, V), lb.reshape(-1),
                reduction="none", ignore_index=-100, label_smoothing=ls,
            ).view(lb.shape)
            valid = (lb != -100).float()
            loss_num = loss_num + (ce * wt * valid).sum()
            loss_den = loss_den + (wt * valid).sum()
            # 监控: 分类/desc loss 分开(张量累积,log 步才 .item(),
            # 避免每步 XLA 同步)
            with torch.no_grad():
                cls_m = (wt > 1.0).float() * valid
                desc_m = (wt == 1.0).float() * valid
                cls_sum = cls_sum + (ce.detach() * cls_m).sum()
                cls_cnt = cls_cnt + cls_m.sum()
                desc_sum = desc_sum + (ce.detach() * desc_m).sum()
                desc_cnt = desc_cnt + desc_m.sum()
        loss = loss_num / loss_den.clamp_min(1.0)

        with torch.no_grad():
            self._loss_log["cls"] += (cls_sum / cls_cnt.clamp_min(1.0))
            self._loss_log["desc"] += (desc_sum / desc_cnt.clamp_min(1.0))
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
        cls_v, desc_v = self._loss_log["cls"], self._loss_log["desc"]
        # 张量累积 → 仅在 log 步同步一次
        logs["loss_cls"] = round(
            (cls_v.item() if hasattr(cls_v, "item") else cls_v) / n, 4)
        logs["loss_desc"] = round(
            (desc_v.item() if hasattr(desc_v, "item") else desc_v) / n, 4)
        self._loss_log = {"cls": 0.0, "desc": 0.0, "n": 0}
        # HBM 观测: 每个 logging 周期打一行(客户反馈"看不到显存"→ 集成进日志)
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            mi = xm.get_memory_info(torch_xla.device())
            gb = 1024 ** 3
            logs["hbm_gb"] = round(mi["bytes_used"] / gb, 2)
            logs["hbm_peak_gb"] = round(mi.get("peak_bytes_used",
                                               mi["bytes_used"]) / gb, 2)
            logs["hbm_limit_gb"] = round(mi["bytes_limit"] / gb, 2)
        except Exception:                        # 非 XLA 环境静默跳过
            pass
        return super().log(logs, *args, **kwargs)
