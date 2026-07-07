"""KTO 偏好优化(自实现;损失数学与 TRL v1.x KTOTrainer 逐项对齐)。

为什么自实现(2026-07 调研,详见 README 偏离说明):
  - TRL KTOTrainer 无 TPU/torch_xla 支持(CI 仅 GPU/CPU,XLA 路径失修),
    且不支持视频输入(v1.6.0 起仅图像)
  - EasyDeL 有 TPU 原生 KTO,但是 JAX 且纯文本管线,偏离已定稿路线 A
  - 双 adapter 做 ref 的设计与 TRL v1.x 官方机制一致(TRL 对 PeftModel
    自动复制 default adapter 为冻结 ref adapter,ref_adapter_name 已删除)

KTO 损失(Ethayarajh et al. 2024;KL 基准用"错配对"估计,同 TRL):
  错配对 = 同 batch 内 (video_i, completion_{i+1})
  (本任务生产 prompt 全局唯一,错配只需滚动 completion 张量)
  kl = mean(logp_policy(错配) - logp_ref(错配)).detach()
       → 跨 TPU 核 all_reduce 平均 → clamp(min=0)
  desirable:   w_d * (1 - sigmoid(beta * (logratio - kl)))
  undesirable: w_u * (1 - sigmoid(beta * (kl - logratio)))

TPU 工程要点(对应 WORK_STATUS 风险清单):
  - sum_logprob 分块计算,不物化 (B,L,V) float32 log_softmax
    (Gemma 词表 ~262k,B8×L2047 全量物化 ≈17GB,v6e 必 OOM)
  - plan_stratified_batches: 每 batch 固定混入 undesirable(错例仅占
    ~10-15%,随机采样会出现大量全 desirable batch,KL/梯度信号失真)
  - ref/policy 同起点,logratio 起步为 0 是预期;ref_divergence_alert
    在 warn_after 步后仍 ≈0 → set_adapter 未生效(静默退化),必须 raise
  - 每步 4 次 forward(policy/ref × 匹配/错配),仅 policy×匹配带梯度;
    gradient_checkpointing 建议开启

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


# ====================== 纯逻辑(torch-free,可单测) ======================

def plan_stratified_batches(is_desirable, batch_size, n_undesirable, rng):
    """分层 batch 计划:每个 batch 固定混入 n_undesirable 条错例。

    is_desirable: List[bool](数据集顺序)。返回 List[List[int]]。
    epoch 长度由 desirable 池决定;undesirable 不足时循环重采
    (错例仅 ~10-15%,复用是有意的 —— KTO 论文允许非配对/不均衡)。
    n_undesirable=0 或无错例时退化为普通 shuffle 分批。
    """
    des = [i for i, d in enumerate(is_desirable) if d]
    und = [i for i, d in enumerate(is_desirable) if not d]
    n_u = min(n_undesirable, batch_size - 1) if und else 0
    if n_u <= 0:
        idx = list(range(len(is_desirable)))
        rng.shuffle(idx)
        return [idx[i:i + batch_size]
                for i in range(0, len(idx) - batch_size + 1, batch_size)]
    rng.shuffle(des)
    rng.shuffle(und)
    n_d = batch_size - n_u
    batches, u_pos = [], 0
    for i in range(0, len(des) - n_d + 1, n_d):
        batch = des[i:i + n_d]
        for _ in range(n_u):
            if u_pos >= len(und):
                rng.shuffle(und)
                u_pos = 0
            batch.append(und[u_pos])
            u_pos += 1
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def ref_divergence_alert(weight_gap, step, warn_after=100, tol=1e-7):
    """True = 训练疑似空转(policy 权重没有离开 reference 起点)。

    输入 = adapter_weight_gap(model)(权重空间 L1 差)。
    ⚠️ 为什么不用输出空间(logratio / KL 差)检测: v6e-1 实测同权重下
    带梯度图 vs no-grad 图差 ~0.15、两个 no-grad 图(XLA 融合调度不同)
    差 ~0.2 —— bf16 图间噪声淹没一切"应为零"的输出信号。
    权重空间无此问题: 起点精确 0,优化器真更新后 > 0。
    """
    return step >= warn_after and weight_gap < tol


def adapter_weight_gap(model, sample_per_kind=4):
    """default vs reference adapter 的权重 L1 差。

    lora_A 与 lora_B 各采样 sample_per_kind 个张量 —— 只采开头几个会
    全部落在 lora_A 上,而小 lr 下 A 的更新可能先于 B 被舍入吞掉,
    导致假零。同步开销小但非零 —— 只在 logging 步调用。
    """
    gap, got = 0.0, {"lora_a": 0, "lora_b": 0}
    sd = model.state_dict()
    for k in sd:
        low = k.lower()
        if ".default." not in k or not ("lora_a" in low or "lora_b" in low):
            continue
        kind = "lora_a" if "lora_a" in low else "lora_b"
        if got[kind] >= sample_per_kind:
            continue
        rk = k.replace(".default.", ".reference.")
        if rk in sd:
            gap += (sd[k].float() - sd[rk].float()).abs().mean().item()
            got[kind] += 1
    return gap / max(sum(got.values()), 1)


def classification_brake(baseline, current, drop_threshold=0.5):
    """KTO 安全刹车(training_plan 10.2): 返回劣化超阈值的指标名列表。

    baseline/current: {"RoleType_acc": 89.5, "SubKeyScene_acc": 81.0, ...}
    (百分制)。非空返回值 → 停止训练,LR 减半重试或放弃保留 v1.5。
    """
    degraded = []
    for k, base in baseline.items():
        cur = current.get(k)
        if cur is not None and cur < base - drop_threshold:
            degraded.append(k)
    return degraded


# ====================== 张量数学(需 torch,可 CPU 对拍) ======================

def sum_logprob(logits, labels, chunk_size=256, window=None):
    """completion 段(labels != -100)的 sum logp。

    window=(start, end): completion 在序列中的静态位置(本任务生产 prompt
    锁死 → 所有样本 prompt token 长度相同,窗口可静态确定)。
    ⚠️ TPU 真机实测(v6e-1): 必须传 window —— 不传时 `logits[:, :-1]`
    会让 XLA 物化一份全长副本(B8×L2048×V262k bf16 ≈ 8.6GB),叠加分块
    临时张量峰值 24GB,32GB HBM 直接 OOM;切窗后峰值仅 ~2GB。
    chunk_size 固定 → 循环步数静态,XLA 只编译一次。
    """
    import torch
    if window is not None:
        w_s = max(int(window[0]), 1)
        w_e = min(int(window[1]), labels.shape[1])   # 越界 clamp,防切片错位
        logits = logits[:, w_s - 1:w_e - 1]
        tgt = labels[:, w_s:w_e]
    else:
        logits = logits[:, :-1]
        tgt = labels[:, 1:]
    mask = (tgt != -100)
    total = logits.new_zeros(logits.shape[0], dtype=torch.float32)
    for s in range(0, tgt.shape[1], chunk_size):
        lg = logits[:, s:s + chunk_size].float()
        t = tgt[:, s:s + chunk_size].clamp_min(0)
        m = mask[:, s:s + chunk_size]
        tok = lg.gather(2, t.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(dim=-1)
        total = total + (tok * m.to(tok.dtype)).sum(dim=1)
    return total


# gemma-4 多模态前向键(v6e-1 烟测确认;pixel_values 键名不适用)
MM_KEYS = ("pixel_values_videos", "video_position_ids", "mm_token_type_ids")


def completion_logprob(model, input_ids, attention_mask, mm_inputs,
                       labels, chunk_size=256, window=None):
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                **mm_inputs)
    return sum_logprob(out.logits, labels, chunk_size, window)


def roll_completions(input_ids, attention_mask, labels):
    """构造错配 KL batch: completion 滚动 1 位、视频不动
    → (video_i, completion_{i+1})。前提: 生产 prompt 对所有样本相同
    (本任务成立 —— prompt 锁死),滚动整条序列即等价于滚动 completion。
    """
    import torch
    return (torch.roll(input_ids, 1, dims=0),
            torch.roll(attention_mask, 1, dims=0),
            torch.roll(labels, 1, dims=0))


def kto_loss(logp_policy, logp_ref, kl_policy, kl_ref, is_desirable,
             beta, w_d, w_u, kl_reduce_fn=None):
    """与 TRL KTOTrainer.kto_loss 数学一致(cat 两段 → torch.where 向量化,
    等价且无空段占位问题)。kl_reduce_fn: 跨设备平均(TPU 传 all_reduce;
    单机/CPU 对拍传 None)。返回 (loss, 监控张量 dict —— 调用方按
    logging 步频再 .item(),避免每步 XLA 同步)。
    """
    import torch
    kl = (kl_policy - kl_ref).mean().detach()
    if kl_reduce_fn is not None:
        kl = kl_reduce_fn(kl)
    kl = kl.clamp(min=0)

    logratio = logp_policy - logp_ref
    d = is_desirable.bool()
    losses = torch.where(
        d,
        w_d * (1 - torch.sigmoid(beta * (logratio - kl))),
        w_u * (1 - torch.sigmoid(beta * (kl - logratio))))
    return losses.mean(), {
        "kl": kl,
        # ref_gap 仅作趋势观察: v6e-1 实测同权重下也有 ~0.2 的 bf16
        # 图间噪声(XLA 融合调度差异),零检测请用 adapter_weight_gap
        "ref_gap": (kl_policy - kl_ref).abs().mean().detach(),
        "logratio_abs_mean": logratio.abs().mean().detach(),
        "reward_desirable": (beta * logratio.detach())[d],
        "reward_undesirable": (beta * logratio.detach())[~d],
    }


def kto_step(model, batch, beta, w_d, w_u, chunk_size=256,
             kl_reduce_fn=None, window=None):
    """4 次 forward: policy/ref × 匹配/错配;仅 policy×匹配带梯度。
    window: completion 静态窗口(prompt 锁死 → 可静态确定),TPU 必传。"""
    import torch
    ids, am = batch["input_ids"], batch["attention_mask"]
    labels, is_desirable = batch["labels"], batch["is_desirable"]
    # 视频张量不滚动(错配 = video_i × completion_{i+1});mm_token_type_ids
    # 的 prompt 段全批相同、target 段全 0,滚动与否等价,保持原样
    mm = {k: batch[k] for k in MM_KEYS if k in batch}
    kl_ids, kl_am, kl_labels = roll_completions(ids, am, labels)

    logp_policy = completion_logprob(model, ids, am, mm, labels,
                                     chunk_size, window)
    with torch.no_grad():
        kl_policy = completion_logprob(model, kl_ids, kl_am, mm,
                                       kl_labels, chunk_size, window)
        model.set_adapter("reference")
        logp_ref = completion_logprob(model, ids, am, mm, labels,
                                      chunk_size, window)
        kl_ref = completion_logprob(model, kl_ids, kl_am, mm,
                                    kl_labels, chunk_size, window)
        model.set_adapter("default")

    return kto_loss(logp_policy, logp_ref, kl_policy, kl_ref,
                    is_desirable, beta, w_d, w_u, kl_reduce_fn)


# ====================== 入口 ======================

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
    # bf16 参数吞小更新(v6e-1 实测),可训张量必须 fp32 master weights
    from training.common import (cast_trainable_to_fp32,
                                 enable_xla_gradient_checkpointing)
    cast_trainable_to_fp32(model)
    # 4 次 forward/步 → 激活显存 ×,checkpointing 必开(XLA 专用版,
    # 原生 torch.utils.checkpoint 会被 CSE 优化掉,烟测确认)
    enable_xla_gradient_checkpointing(model)
    model.train()   # checkpoint 分支仅 training 模式生效(烟测踩坑)

    import torch
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    model.to(device)

    kto_records = [json.loads(l) for l in open(kcfg["kto_data"],
                                               encoding="utf-8")]
    n_und = sum(1 for r in kto_records if not r["label"])
    print(f"[kto] {len(kto_records)} samples ({n_und} undesirable)")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(kcfg["learning_rate"]))

    def kl_all_reduce(t):
        # SMOKE: torch_xla 新旧 API 二选一(xr.world_size / xm.xrt_world_size)
        try:
            from torch_xla import runtime as xr
            ws = xr.world_size()
        except Exception:
            ws = xm.xrt_world_size()
        return xm.all_reduce(xm.REDUCE_SUM, t) / max(ws, 1)

    collator = AnkerCollator(processor, cfg)
    # ---- 主循环骨架(DataLoader 拼装依赖 processor 真实签名,烟测时收尾):
    # import random
    # rng = random.Random(0)
    # batches = plan_stratified_batches(
    #     [r["label"] for r in kto_records],
    #     cfg["train"]["per_device_batch_size"],
    #     kcfg.get("undesirable_per_batch", 2), rng)
    # baseline = <v1.5 在监控集上的分类指标>
    # for step, idx in enumerate(batches):   # KTOVideoDataset + collator 出 batch
    #     batch["is_desirable"] = ...        # 由 kto_data 的 label 字段构造
    # import numpy as np
    # dummy = np.zeros((16, 384, 384, 3), np.uint8)
    # p_len = collator._encode_prompt(dummy, "")["input_ids"].shape[1]
    #                                            # prompt 锁死 → 静态长度
    # window = (p_len, p_len + kcfg.get("completion_budget", 256))
    #     loss, logs = kto_step(model, batch, kcfg["beta"],
    #                           kcfg["desirable_weight"],
    #                           kcfg["undesirable_weight"],
    #                           kcfg.get("logprob_chunk", 256), kl_all_reduce,
    #                           window)   # TPU 必传,否则全长 logits 副本 OOM
    #     loss.backward(); xm.optimizer_step(opt); opt.zero_grad()
    #     if step % logging_steps == 0:
    #         gap = adapter_weight_gap(model)     # 权重空间,起点精确 0
    #         if ref_divergence_alert(gap, step,
    #                 kcfg.get("ref_divergence_warn_after", 100)):
    #             raise RuntimeError("policy 权重未离开 reference 起点 — "
    #                                "优化器/梯度流失效,训练在空转")
    #     if step % 100 == 0:
    #         current = <监控集评测>
    #         bad = classification_brake(baseline, current)
    #         if bad: <停止;LR 减半重试或放弃保留 v1.5>
    raise SystemExit(
        "SMOKE: KTO 主循环骨架已就绪(损失/分块 logprob/分层 batch/告警"
        "已实现并有测试),DataLoader 拼装在 TPU 烟测时完成"
        "(依赖 processor 真实入参签名,见 WORK_STATUS 烟测清单)")


if __name__ == "__main__":
    main()
