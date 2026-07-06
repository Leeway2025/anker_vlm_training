"""KTO 偏好优化(自实现,LoRA disable_adapter 做参考模型 —— 免 TRL 的
VLM 兼容性问题,且零额外显存放 ref model)。

KTO 损失(Ethayarajh et al. 2024,分类偏好版):
  logratio = logp_policy(completion) - logp_ref(completion)
  z0       = batch 内 logratio 均值(detach, clamp≥0)—— KL 基准
  desirable:   w_d * (1 - sigmoid(beta * (logratio - z0)))
  undesirable: w_u * (1 - sigmoid(beta * (z0 - logratio)))

ref = 同一模型 disable_adapter()(即 Phase 5d 起点的 base+此前 LoRA?
     不 —— disable_adapter 得到的是"纯 base"。KTO 的 ref 应为 SFT 后模型,
     因此做法: 训前把 v1.5 adapter 权重拷贝一份冻结为 "reference" adapter,
     policy 用 "default" adapter(从 v1.5 初始化,可训)。
     peft 多 adapter 切换零显存开销。)

启动:
  python -m torch_xla.distributed.xla_spawn --num_cores 8 \
      training/kto.py --config configs/phase6_kto.yaml
"""
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def completion_logprob(model, input_ids, attention_mask, pixel_values,
                       labels):
    """sum logp over completion tokens(labels != -100 的位置)。"""
    import torch
    import torch.nn.functional as F
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values)
    logits = out.logits[:, :-1].float()
    tgt = labels[:, 1:]
    mask = (tgt != -100)
    logp = F.log_softmax(logits, dim=-1)
    tok_logp = torch.gather(
        logp, 2, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum(dim=1)


def kto_step(model, batch, beta, w_d, w_u):
    import torch
    ids, am, pv = batch["input_ids"], batch["attention_mask"], \
        batch["pixel_values"]
    labels, is_desirable = batch["labels"], batch["is_desirable"]

    logp_policy = completion_logprob(model, ids, am, pv, labels)
    model.set_adapter("reference")
    with torch.no_grad():
        logp_ref = completion_logprob(model, ids, am, pv, labels)
    model.set_adapter("default")

    logratio = logp_policy - logp_ref
    z0 = logratio.detach().mean().clamp_min(0)

    d = is_desirable.bool()
    loss_d = w_d * (1 - torch.sigmoid(beta * (logratio[d] - z0))) \
        if d.any() else logratio.new_zeros(1)
    loss_u = w_u * (1 - torch.sigmoid(beta * (z0 - logratio[~d]))) \
        if (~d).any() else logratio.new_zeros(1)
    loss = torch.cat([loss_d, loss_u]).mean()
    return loss, {"logratio_mean": logratio.mean().item(),
                  "z0": z0.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", default=None,
                    help="覆盖 config 的 init_from(pipeline 缝合用)")
    ap.add_argument("--output", default="outputs/phase6_kto")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    kcfg = yaml.safe_load(open(a.config, encoding="utf-8"))

    from training.common import load_model_and_processor, freeze_base, build_lora
    from data.build_dataset import AnkerCollator
    from data.formatting import build_target  # noqa: F401(collator 内已处理)

    model, processor = load_model_and_processor(cfg)
    freeze_base(model, cfg)
    model = build_lora(model, cfg)
    init = a.init_from or kcfg["init_from"]
    # default 已由 build_lora 注入 → 用 state_dict 灌权重(避免名字冲突)
    from peft import load_peft_weights, set_peft_model_state_dict
    set_peft_model_state_dict(model, load_peft_weights(init))
    # reference = v1.5 副本,冻结(policy/ref 同起点)
    model.load_adapter(init, adapter_name="reference")
    for n, p in model.named_parameters():
        if ".reference." in n:
            p.requires_grad = False
    model.set_adapter("default")
    # projector: 加载 v1.5 权重并冻结(KTO 只动 LoRA)
    import torch as _t, os as _os
    pj = _os.path.join(init, "projector.pt")
    if _os.path.exists(pj):
        model.load_state_dict(_t.load(pj, map_location="cpu"), strict=False)
        print(f"[kto] projector loaded from {pj} (frozen)")
    else:
        print(f"[WARN] {init} 无 projector.pt — v1.5 projector 成果缺失!")
    for n, p in model.named_parameters():
        if any(kw in n.lower() for kw in cfg["freeze"]["projector_keywords"]) \
                and "lora" not in n.lower():
            p.requires_grad = False

    # ---- KTO 数据: completion 已是完整目标串,构造与 SFT 相同的张量 ----
    # 简化: 复用 AnkerCollator 的 prompt 编码逻辑,labels 全 completion 段
    # (KTO 无 think/权重需求,token_weights 忽略)
    # 训练循环: 标准 torch_xla 循环(略去 Trainer,控制 set_adapter 时机)
    import torch
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    model.to(device)

    kto_records = [json.loads(l) for l in open(kcfg["kto_data"],
                                               encoding="utf-8")]
    print(f"[kto] {len(kto_records)} samples")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(kcfg["learning_rate"]))

    collator = AnkerCollator(processor, cfg)
    # DataLoader 构造(帧解码走 AnkerVideoDataset 同款路径)略:
    # 与 train.py 共用 KTOVideoDataset(见 REPRODUCE.md 数据格式)
    # —— 主循环骨架:
    # for step, batch in enumerate(loader):
    #     batch["is_desirable"] = batch.pop("kto_label")
    #     loss, log = kto_step(model, batch, kcfg["beta"],
    #                          kcfg["desirable_weight"],
    #                          kcfg["undesirable_weight"])
    #     loss.backward(); xm.optimizer_step(opt); opt.zero_grad()
    #     每 100 step: 监控集评测,任一分类指标降 >0.5 → 停,LR 减半
    raise SystemExit(
        "SMOKE: KTO 主循环骨架已就绪,DataLoader 拼装在 TPU 烟测时完成 "
        "(依赖 processor 真实入参签名,见 WORK_STATUS 烟测清单)")


if __name__ == "__main__":
    main()
