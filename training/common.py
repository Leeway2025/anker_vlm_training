"""模型装载 / 冻结 / LoRA / 优化器 / 辅助头(需 GPU 环境烟测的模块)。

多 LoRA 部署红线(training_plan 5.3 节):
  base 主干 / PLE / Embedding 绝对冻结 —— freeze_base 里有数量断言,
  冻结关键字没匹配到任何参数时直接报错(防止 gemma-4 命名变化悄悄失效)。
"""
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.taxonomy import AUX_VOCABS, AUX_HEAD_ORDER, KS_CLASSES  # noqa: E402


def load_model_and_processor(cfg):
    import torch
    from transformers import AutoProcessor
    name = cfg["model"]["name_or_path"]
    processor = AutoProcessor.from_pretrained(name)
    dtype = getattr(torch, cfg["model"]["torch_dtype"])
    kwargs = dict(torch_dtype=dtype)
    try:
        kwargs["attn_implementation"] = cfg["model"]["attn_implementation"]
        model = _load(name, **kwargs)
    except Exception:
        kwargs["attn_implementation"] = "sdpa"
        model = _load(name, **kwargs)
    return model, processor


def _load(name, **kw):
    # SMOKE: gemma-4 的注册类名需真机确认,依次尝试
    from transformers import AutoModelForImageTextToText
    try:
        return AutoModelForImageTextToText.from_pretrained(name, **kw)
    except Exception:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True, **kw)


def freeze_base(model, cfg):
    """冻结全部参数;LoRA adapter 由 peft 注入时自带 requires_grad=True;
    Projector 全参放开。返回统计 dict(写进训练日志,人工核对)。"""
    fz = cfg["freeze"]
    for p in model.parameters():
        p.requires_grad = False

    frozen_hits = {k: 0 for k in fz["keywords"]}
    proj_params = 0
    for name, p in model.named_parameters():
        low = name.lower()
        for kw in fz["keywords"]:
            if kw in low:
                frozen_hits[kw] += 1          # 已冻结,只计数验证命名存在
        if any(k in low for k in fz["projector_keywords"]):
            p.requires_grad = True            # Vision Projector 全参
            proj_params += p.numel()

    # 断言: PLE / embed 关键字必须命中(gemma-4 命名变化时立刻暴露)
    missed = [k for k in ("embed_tokens",) if frozen_hits.get(k, 0) == 0]
    if missed:
        raise RuntimeError(
            f"freeze keywords not found in model params: {missed} — "
            f"检查 gemma-4 参数命名,更新 configs/base.yaml freeze.keywords")
    if frozen_hits.get("ple", 0) == 0 and \
       frozen_hits.get("per_layer_embedding", 0) == 0:
        print("[WARN] no PLE-named params found — 确认 gemma-4 PLE 命名"
              "(可能叫 per_layer_*),否则多 LoRA 红线可能失守")
    if proj_params == 0:
        raise RuntimeError("projector keywords matched nothing — "
                           "检查 projector_keywords 配置")
    return {"frozen_keyword_hits": frozen_hits,
            "projector_trainable_params": proj_params}


def detect_global_layers(model):
    """从 config.layer_types 找 full-attention 层索引(差异化 rank 用)。
    gemma-4 e4b config: 5×sliding + 1×full 重复;找不到时按该模式推断。"""
    cfg = model.config
    tc = getattr(cfg, "text_config", cfg)
    lt = getattr(tc, "layer_types", None)
    if lt:
        return [i for i, t in enumerate(lt)
                if "full" in str(t) or "global" in str(t)]
    n = getattr(tc, "num_hidden_layers", 35)
    print(f"[WARN] config.layer_types missing — 按 5:1 模式推断 global 层")
    return [i for i in range(n) if i % 6 == 5]


def build_lora(model, cfg):
    """差异化 rank + rsLoRA + PISSA(失败回退)。"""
    from peft import LoraConfig, get_peft_model
    lcfg = cfg["lora"]
    targets = sorted(set(lcfg["llm_targets"]) | set(lcfg["vision_targets"]))
    glb = detect_global_layers(model)

    rank_pattern, alpha_pattern = {}, {}
    for i in glb:
        # 匹配 …layers.{i}.<any>.{proj};结尾锚定防 5 匹配 51
        for t in lcfg["llm_targets"]:
            key = rf".*\.layers\.{i}\..*\.{t}"
            rank_pattern[key] = lcfg["r_global"]
            alpha_pattern[key] = int(lcfg["r_global"] * lcfg["alpha_ratio"])

    base_kwargs = dict(
        r=lcfg["r_sliding"],
        lora_alpha=int(lcfg["r_sliding"] * lcfg["alpha_ratio"]),
        target_modules=targets,
        lora_dropout=lcfg["dropout"],
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        use_rslora=lcfg["use_rslora"],
        task_type="CAUSAL_LM",
    )
    try:
        peft_model = get_peft_model(model, LoraConfig(
            init_lora_weights=lcfg["init_weights"], **base_kwargs))
        print(f"[LoRA] init={lcfg['init_weights']}, global layers={glb}")
    except Exception as e:
        print(f"[WARN] init_weights={lcfg['init_weights']} failed ({e}); "
              f"fallback to default init")
        peft_model = get_peft_model(model, LoraConfig(**base_kwargs))
    return peft_model


def build_optimizer(model, cfg, aux_module=None, lr_scale=1.0):
    """参数分组: LoRA+ (B×16) / vision vs llm / projector / aux heads。"""
    import torch
    lrs = cfg["lr"]
    ratio = cfg["lora"]["loraplus_lr_ratio"]
    vis_kw = cfg["freeze"]["vision_keywords"]
    proj_kw = cfg["freeze"]["projector_keywords"]

    groups = {"llm_A": [], "llm_B": [], "vis_A": [], "vis_B": [],
              "proj": [], "aux": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = name.lower()
        is_vis = any(k in low for k in vis_kw)
        if "lora_a" in low:
            groups["vis_A" if is_vis else "llm_A"].append(p)
        elif "lora_b" in low:
            groups["vis_B" if is_vis else "llm_B"].append(p)
        elif any(k in low for k in proj_kw):
            groups["proj"].append(p)
        else:
            groups["proj"].append(p)     # 其余可训参数按 projector 处理
    if aux_module is not None:
        groups["aux"] = list(aux_module.parameters())

    s = lr_scale
    param_groups = [
        {"params": groups["llm_A"], "lr": lrs["llm_lora"] * s},
        {"params": groups["llm_B"], "lr": lrs["llm_lora"] * ratio * s},
        {"params": groups["vis_A"], "lr": lrs["vision_lora"] * s},
        {"params": groups["vis_B"], "lr": lrs["vision_lora"] * ratio * s},
        {"params": groups["proj"], "lr": lrs["projector"] * s},
        {"params": groups["aux"], "lr": lrs["aux_heads"] * s},
    ]
    param_groups = [g for g in param_groups if g["params"]]

    # TPU-only: 标准 AdamW(bitsandbytes 不支持 XLA,不提供 GPU 路径)
    return torch.optim.AdamW(param_groups,
                             weight_decay=cfg["train"]["weight_decay"])


class AuxHeads:
    """7 个属性辅助头 + attention pooling,挂在 Vision Projector 输出上。
    推理/导出前调用 export/split_deliverables.py 时物理丢弃。"""

    def __init__(self, cfg):
        import torch.nn as nn
        self.nn = nn
        self.heads = None            # lazy init(等首个 hook 拿到 dim)
        self.pool_score = None
        self._feat = None

    def attach(self, model, projector_keywords):
        target = None
        for name, mod in model.named_modules():
            if any(k in name.lower() for k in projector_keywords):
                target = mod         # 取最后一个匹配(最外层 projector)
        if target is None:
            raise RuntimeError("projector module not found for aux hook")
        target.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inp, out):
        self._feat = out if not isinstance(out, tuple) else out[0]

    def _lazy_init(self, dim, device, dtype):
        nn = self.nn
        self.pool_score = nn.Linear(dim, 1).to(device, dtype)
        self.heads = nn.ModuleDict({
            h: nn.Linear(dim, len(v)).to(device, dtype)
            for h, v in AUX_VOCABS.items()})

    def parameters(self):
        if self.heads is None:
            return []
        return list(self.pool_score.parameters()) + \
            list(self.heads.parameters())

    def compute_loss(self, aux_labels):
        """aux_labels: (B, 7) long,-100 忽略。返回 scalar loss(无有效标签时 0)。"""
        import torch
        import torch.nn.functional as F
        if self._feat is None:
            return torch.tensor(0.0)
        f = self._feat                       # (B*, N, D) 或 (B, N, D)
        if f.dim() == 2:
            f = f.unsqueeze(0)
        B = aux_labels.shape[0]
        f = f.reshape(B, -1, f.shape[-1])    # 帧维并入 token 维
        if self.heads is None:
            self._lazy_init(f.shape[-1], f.device, f.dtype)
        attn = torch.softmax(self.pool_score(f).squeeze(-1), dim=-1)
        pooled = torch.einsum("bn,bnd->bd", attn, f)
        loss, n = 0.0, 0
        for j, h in enumerate(AUX_HEAD_ORDER):
            lab = aux_labels[:, j]
            if (lab != -100).any():
                loss = loss + F.cross_entropy(
                    self.heads[h](pooled).float(), lab, ignore_index=-100)
                n += 1
        self._feat = None
        return loss / max(n, 1)


class KSParentHead:
    """KeyScene 6 大类父类头(Phase 5 轻量辅助)。"""

    def __init__(self):
        self.head = None

    def compute_loss(self, hidden_states, labels_mask, ks_labels):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        if (ks_labels == -100).all():
            return torch.tensor(0.0, device=hidden_states.device)
        if self.head is None:
            self.head = nn.Linear(hidden_states.shape[-1],
                                  len(KS_CLASSES)).to(
                hidden_states.device, hidden_states.dtype)
        m = labels_mask.unsqueeze(-1).to(hidden_states.dtype)   # 目标段位置
        pooled = (hidden_states * m).sum(1) / m.sum(1).clamp_min(1)
        return F.cross_entropy(self.head(pooled).float(), ks_labels,
                               ignore_index=-100)

    def parameters(self):
        return [] if self.head is None else list(self.head.parameters())
