"""训练入口(torch_xla / TPU v6e-8)。

启动(8 核):
  PJRT_DEVICE=TPU python -m torch_xla.distributed.xla_spawn --num_cores 8 \
      training/train.py --phase configs/phase5_sft.yaml --stage b \
      [--init-from DIR] [--sample-weights sw.json] [--output DIR]

阶段串联约定(每个 phase 的 final/ 目录 = 完整可续训检查点):
  final/
    adapter_model.safetensors + adapter_config.json   (LoRA;stage a 无)
    projector.pt                                       (Vision Projector 全参)
    aux_heads.pt                                       (仅 Phase 5b)
  下一 phase 通过 init_from 指向上一 phase 的 final/,
  train.py 同时恢复 adapter 权重(set_peft_model_state_dict,避免
  load_adapter 的 default 名冲突)和 projector.pt(strict=False)。

Phase 流水:
  5_sft stage a(仅 projector)→ stage b(--init-from …_a/final)
  → hard_mining(样本物理复制 ×N,XLA 分布式安全,不用 WeightedSampler)
  → 5b_aux → 5d_cot → build_kto_data → kto.py → swa.py
"""
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.build_dataset import (AnkerVideoDataset, AnkerCollator,   # noqa: E402
                                load_jsonl, split_by_camera)
from training.common import (load_model_and_processor, freeze_base,  # noqa: E402
                             build_lora, build_optimizer,
                             AuxHeads, KSParentHead)
from training.trainer import WeightedSFTTrainer                      # noqa: E402


def load_cfg(phase_yaml):
    base = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    phase = yaml.safe_load(open(phase_yaml, encoding="utf-8"))
    return base, phase


def load_records(cfg, phase):
    records = [json.loads(l) for l in
               open(cfg["data"]["labels_file"], encoding="utf-8")]
    if phase.get("use_whitelist_only"):
        wl = set(open(cfg["data"]["whitelist_file"],
                      encoding="utf-8").read().split())
        records = [r for r in records if r["video_id"] in wl]
        print(f"[data] whitelist filter: {len(records)} samples")
    return records


def apply_hard_mining(records, sample_weights: dict):
    """错例按权重物理复制(权重 3.0 → 出现 3 次)。
    为什么不用 WeightedRandomSampler: XLA 多核下 Trainer 会套
    DistributedSampler,二者不能组合;物理复制在任何 sampler 下都成立。"""
    out = []
    for r in records:
        n = max(1, round(sample_weights.get(r["video_id"], 1.0)))
        out.extend([r] * n)
    print(f"[hard-mining] {len(records)} -> {len(out)} samples")
    return out


def restore_from(model, init_dir, inject_lora):
    """跨 phase 恢复: adapter 权重 + projector 权重。"""
    import torch
    if init_dir is None or not os.path.isdir(init_dir):
        if init_dir:
            raise FileNotFoundError(f"init_from dir not found: {init_dir}")
        return
    ad = os.path.join(init_dir, "adapter_model.safetensors")
    if inject_lora and os.path.exists(ad):
        # 不用 load_adapter("default")——default 已由 build_lora 注入,会冲突
        from peft import load_peft_weights, set_peft_model_state_dict
        sd = load_peft_weights(init_dir)
        result = set_peft_model_state_dict(model, sd)
        unexpected = getattr(result, "unexpected_keys", [])
        print(f"[init] adapter weights loaded from {init_dir} "
              f"(unexpected={len(unexpected)})")
    pj = os.path.join(init_dir, "projector.pt")
    if os.path.exists(pj):
        missing, unexpected = model.load_state_dict(
            torch.load(pj, map_location="cpu"), strict=False)
        print(f"[init] projector loaded from {pj}")
    elif inject_lora and os.path.exists(ad):
        print(f"[WARN] {init_dir} 无 projector.pt — Phase 5 的 projector "
              f"成果不会被继承,确认这是有意为之")


def save_final(model, out_dir, cfg, aux, inject_lora):
    import torch
    final = os.path.join(out_dir, "final")
    os.makedirs(final, exist_ok=True)
    if inject_lora:
        model.save_pretrained(final)        # 仅 adapter(peft 行为)
    # projector 始终单独保存(全参训练,不在 adapter 内)
    proj_sd = {k: v.cpu() for k, v in model.state_dict().items()
               if any(kw in k.lower()
                      for kw in cfg["freeze"]["projector_keywords"])
               and "lora" not in k.lower()}
    torch.save(proj_sd, os.path.join(final, "projector.pt"))
    if aux is not None and aux.heads is not None:
        torch.save({"pool": aux.pool_score.state_dict(),
                    "heads": aux.heads.state_dict()},
                   os.path.join(final, "aux_heads.pt"))
    print(f"[save] {final} (adapter={inject_lora}, "
          f"projector tensors={len(proj_sd)})")
    return final


class AnnealCallback:
    """Phase 5d: 最后 anneal_epochs 切纯生产模式(防 think 泄漏)。"""

    def __init__(self, dataset, total_steps, anneal_epochs, epochs):
        self.ds = dataset
        self.switch_at = int(total_steps * (1 - anneal_epochs / epochs))
        self.done = False

    def on_step_end(self, args, state, control, **kw):
        if not self.done and state.global_step >= self.switch_at:
            self.ds.set_anneal(True)
            self.done = True
            print(f"[anneal] step {state.global_step}: 切换纯生产模式")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--stage", choices=["a", "b"], default="b")
    ap.add_argument("--init-from", default=None,
                    help="覆盖 phase yaml 的 init_from(hard mining 续训必传)")
    ap.add_argument("--sample-weights", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--epochs", type=float, default=None,
                    help="覆盖 phase yaml 的 epochs(追加训练: "
                         "--init-from <本阶段 final> --epochs 2)")
    ap.add_argument("--resume", default=None,
                    help="从训练中断的 checkpoint 目录续跑(透传 Trainer)")
    a = ap.parse_args()

    cfg, phase = load_cfg(a.phase)
    is_phase5 = phase["phase"].startswith("5_sft")
    inject_lora = not (is_phase5 and a.stage == "a")   # stage a 仅 projector
    suffix = f"_{a.stage}" if is_phase5 else ""
    out_dir = a.output or f"outputs/{phase['phase']}{suffix}"
    os.makedirs(out_dir, exist_ok=True)
    init_dir = a.init_from or phase.get("init_from")

    # ---- 数据 ----
    records = load_records(cfg, phase)
    train_recs, val_recs = split_by_camera(
        records, cfg["data"]["val_size"], cfg["data"]["val_holdout_key"])
    if a.sample_weights:
        train_recs = apply_hard_mining(
            train_recs, json.load(open(a.sample_weights)))
    attributes = load_jsonl(cfg["data"]["attributes_file"]) \
        if phase.get("enable_aux_heads") else {}
    reasoning = load_jsonl(cfg["data"]["reasoning_file"]) \
        if phase.get("cot_mode") else {}

    train_ds = AnkerVideoDataset(train_recs, cfg, phase, training=True,
                                 attributes=attributes, reasoning=reasoning)
    val_ds = AnkerVideoDataset(val_recs[:2000], cfg, phase, training=False,
                               attributes=attributes, reasoning=reasoning)

    # ---- 模型: load → freeze → (LoRA) → restore 上一 phase ----
    model, processor = load_model_and_processor(cfg)
    stats = freeze_base(model, cfg)
    print(f"[freeze] {stats}")
    if inject_lora:
        model = build_lora(model, cfg)
    restore_from(model, init_dir, inject_lora)

    aux = None
    if phase.get("enable_aux_heads"):
        aux = AuxHeads(cfg).attach(model, cfg["freeze"]["projector_keywords"])
    ks = KSParentHead() if phase.get("enable_ks_parent_head") else None

    optimizer = build_optimizer(model, cfg, aux_module=aux,
                                lr_scale=phase.get("learning_rate_scale", 1.0))

    # ---- Trainer ----
    from transformers import TrainingArguments
    epochs = a.epochs or phase.get("epochs") or (
        phase["epochs_stage_a"] if a.stage == "a" else phase["epochs_stage_b"])
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["train"]["per_device_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation"],
        warmup_steps=cfg["train"]["warmup_steps"],
        max_grad_norm=cfg["train"]["max_grad_norm"],
        bf16=cfg["train"]["bf16"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        eval_steps=cfg["train"]["eval_steps"],
        eval_strategy="steps",
        save_total_limit=3,                 # SWA 需要最后 3 个 checkpoint
        # epochs 是上限;早停由验证集决定实际长度
        load_best_model_at_end=cfg["train"].get("load_best_at_end", True),
        metric_for_best_model="eval_loss",  # ×4 加权 → 分类偏重信号
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=cfg["data"]["num_workers"],
        report_to="tensorboard",
    )
    from transformers import EarlyStoppingCallback
    trainer = WeightedSFTTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=AnkerCollator(processor, cfg),
        optimizers=(optimizer, None),
        run_cfg=cfg, ks_head=ks, aux_heads=aux,
    )
    pat = cfg["train"].get("early_stopping_patience", 0)
    if pat:
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=pat))
    if phase.get("cot_mode"):
        total = len(train_ds) * epochs // (
            cfg["train"]["per_device_batch_size"] *
            cfg["train"]["gradient_accumulation"])
        trainer.add_callback(AnnealCallback(
            train_ds, total, phase.get("cot_anneal_epochs", 0.5), epochs))

    trainer.train(resume_from_checkpoint=a.resume)
    # load_best_model_at_end=True → 此刻 model 已是验证集最优权重
    save_final(model, out_dir, cfg, aux, inject_lora)


if __name__ == "__main__":
    main()
