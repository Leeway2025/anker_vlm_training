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


def strip_audio_modules(model):
    """剥离 audio_tower / embed_audio(本任务无音频输入,forward 的音频
    分支永不触发,但参数常驻 HBM)。省显存 + 加速加载;
    注意: 剥离后的进程内 state_dict 不含 audio 键,跨阶段 restore 均
    strict=False,不受影响;交付物(llm_adapter/vision_merged)本就不含 audio。"""
    n = 0
    for _, mod in list(model.named_modules()):
        for child in ("audio_tower", "embed_audio"):
            sub = getattr(mod, child, None)
            if sub is not None and hasattr(sub, "parameters"):
                n += sum(p.numel() for p in sub.parameters())
                setattr(mod, child, None)
    if n:
        print(f"[strip] audio 模块已剥离: {n/1e6:.0f}M 参数"
              f"(bf16 ≈ {n*2/1e9:.2f}GB HBM)")
    return model


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
    if cfg["model"].get("strip_audio", True):
        strip_audio_modules(model)
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
    ple_keys = ("per_layer", "ple", "per_layer_embedding")
    if not any(frozen_hits.get(k, 0) for k in ple_keys):
        print("[WARN] no PLE-named params found — 确认 gemma-4 PLE 命名"
              "(实测为 per_layer_input_gate/per_layer_projection),"
              "否则多 LoRA 红线可能失守")
    if proj_params == 0:
        raise RuntimeError("projector keywords matched nothing — "
                           "检查 projector_keywords 配置")
    return {"frozen_keyword_hits": frozen_hits,
            "projector_trainable_params": proj_params}


def enable_xla_gradient_checkpointing(model):
    """XLA 专用 gradient checkpointing。

    坑(v6e-1 真机实测): torch.utils.checkpoint 的重算子图会被 XLA 的
    CSE 优化合并回去 —— 显存毫无节省(B2/L2048 视频反向 35.4G OOM,
    开关前后字节不差)。必须用 torch_xla.utils.checkpoint,它插入
    optimization_barrier 阻止 CSE。
    做法: 把 torch.utils.checkpoint.checkpoint 打补丁为 XLA 版
    (transformers 的 _gradient_checkpointing_func 在 enable 时解析该符号),
    并 enable_input_require_grads(冻结 embedding + LoRA 下 reentrant
    checkpoint 需要输入侧有梯度流)。
    """
    try:
        import torch_xla.utils.checkpoint as xla_ckpt
        _xla_checkpoint = xla_ckpt.checkpoint

        def _ckpt(fn, *args, use_reentrant=None, **kw):
            return _xla_checkpoint(fn, *args)

        # transformers.modeling_utils 在模块加载时 `from torch.utils.checkpoint
        # import checkpoint` 早绑定,补丁必须打在它自己的符号上
        # (只改 torch.utils.checkpoint.checkpoint 无效 —— v6e-1 实测
        #  显存字节不差)
        import transformers.modeling_utils as _mu
        _mu.checkpoint = _ckpt
        import torch.utils.checkpoint as _tuc
        _tuc.checkpoint = _ckpt          # 兜底: 其他调用点
        print("[ckpt] XLA checkpoint patched into transformers.modeling_utils")
    except ImportError:
        print("[WARN] torch_xla 不可用,gradient checkpointing 走原生实现")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # ⚠️ v6e-1 实测坑: reentrant checkpoint 在"分段输入不带梯度"时静默
    # 丢梯度。enable_input_require_grads 只覆盖文本 embedding;vision
    # tower 的输入来自冻结 patch_embedder → vision LoRA 的 grad 全为
    # None(llm 侧正常)。补救: 给 vision 侧 embedding 模块输出挂
    # require-grad 钩子,让视觉塔的 checkpoint 段有梯度入口。
    import torch

    def _require_grad_hook(mod, args, output):
        if torch.is_tensor(output) and torch.is_floating_point(output) \
                and not output.requires_grad:
            output.requires_grad_(True)
        return output

    hooked = []
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1].lower()
        if "vision" in name.lower() and "embed" in leaf:
            mod.register_forward_hook(_require_grad_hook)
            hooked.append(name)
    if hooked:
        print(f"[ckpt] vision input-require-grad hooks: {len(hooked)}")


def cast_trainable_to_fp32(model, keyword="lora"):
    """LoRA 参数升 fp32 做 master weights(安全网;peft 新版默认已 fp32)。

    v6e-1 真机实测的两个约束:
    ① bf16 参数 + AdamW(lr≤1e-4)的更新量低于 bf16 舍入分辨率 →
      权重逐字节不变,训练静默空转 —— LoRA(lr 1e-4/2e-5)必须 fp32。
    ② Projector(embed_vision)不能升 fp32: 其 fp32 输出与 bf16
      inputs_embeds 在视频特征 masked_scatter 处相遇,XLA 直接报
      "mixed precision is disallowed";其 lr=5e-4 远超 bf16 分辨率,
      保持 bf16 更新不丢失。
    必须在 restore/注入完成后、建 optimizer 前调用。
    """
    import torch
    n = 0
    for name, p in model.named_parameters():
        if p.requires_grad and p.dtype == torch.bfloat16 \
                and keyword in name.lower():
            p.data = p.data.float()
            n += 1
    print(f"[fp32] {n} LoRA tensors upcast to float32 (master weights)")
    return model


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
    """差异化 rank + rsLoRA + PISSA(失败回退)。

    target 用正则(v6e-1 烟测确认的 gemma-4 结构):
      - LLM 侧 *_proj 是纯 nn.Linear,直接注入
      - vision 侧 *_proj 是 Gemma4ClippableLinear 包装(PEFT 不支持),
        注入点必须是其内部 `.linear` 子模块
      - audio_tower / embed_audio 不在正则内,天然排除
    """
    from peft import LoraConfig, get_peft_model
    lcfg = cfg["lora"]
    glb = detect_global_layers(model)

    llm_alt = "|".join(lcfg["llm_targets"])
    vis_alt = "|".join(lcfg["vision_targets"])
    target_regex = (rf"(.*language_model.*\.({llm_alt}))"
                    rf"|(.*vision_tower.*\.({vis_alt})\.linear)")

    rank_pattern, alpha_pattern = {}, {}
    for i in glb:
        # 匹配 …layers.{i}.<any>.{proj};结尾锚定防 5 匹配 51,
        # 也天然不命中 vision 的 …q_proj.linear(结尾是 linear)
        for t in lcfg["llm_targets"]:
            key = rf".*\.layers\.{i}\..*\.{t}"
            rank_pattern[key] = lcfg["r_global"]
            alpha_pattern[key] = int(lcfg["r_global"] * lcfg["alpha_ratio"])

    base_kwargs = dict(
        r=lcfg["r_sliding"],
        lora_alpha=int(lcfg["r_sliding"] * lcfg["alpha_ratio"]),
        target_modules=target_regex,
        lora_dropout=lcfg["dropout"],
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        use_rslora=lcfg["use_rslora"],
        task_type="CAUSAL_LM",
    )
    # 多 LoRA 红线哨兵: 注入前采样 base 权重,注入后必须逐字节不变
    # (PISSA 会就地改 base → 端侧共享 base 失效;此断言防止任何
    #  init 方式静默突破红线)
    import torch
    probes = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("q_proj.weight") and "language" in name:
                probes.append((name, p.flatten()[:64].clone()))
                if len(probes) >= 3:
                    break
    try:
        peft_model = get_peft_model(model, LoraConfig(
            init_lora_weights=lcfg["init_weights"], **base_kwargs))
        print(f"[LoRA] init={lcfg['init_weights']}, global layers={glb}")
    except Exception as e:
        print(f"[WARN] init_weights={lcfg['init_weights']} failed ({e}); "
              f"fallback to default init")
        peft_model = get_peft_model(model, LoraConfig(**base_kwargs))

    # 红线断言: base 权重未被注入过程修改
    # (包装后键名变为 …q_proj.base_layer.weight,按 stem 匹配)
    with torch.no_grad():
        pd = dict(peft_model.named_parameters())
        for name, snap in probes:
            stem = name[:-len(".weight")]
            cur = next((v for k, v in pd.items()
                        if k.endswith(stem + ".base_layer.weight")
                        or k.endswith(name)), None)
            if cur is None:
                print(f"[WARN] base probe not found after wrap: {name}")
                continue
            if not torch.equal(cur.flatten()[:64].cpu(), snap.cpu()):
                raise RuntimeError(
                    f"base 权重在 LoRA 注入时被修改({name})— 多 LoRA "
                    f"部署红线失守。init_weights 不能用 pissa/olora/corda "
                    f"这类会改 base 的初始化")

    # 审计: LoRA 不得落在排除关键字模块上(audio 等)
    excl = lcfg.get("exclude_keywords", [])
    if excl:
        bad = [n for n, _ in peft_model.named_parameters()
               if "lora" in n.lower()
               and any(k in n.lower() for k in excl)]
        if bad:
            raise RuntimeError(f"LoRA 注入了排除模块: {bad[:3]}")
    n_vis = sum(1 for n, _ in peft_model.named_parameters()
                if "lora" in n.lower() and "vision" in n.lower())
    n_llm = sum(1 for n, _ in peft_model.named_parameters()
                if "lora" in n.lower() and "language" in n.lower())
    if n_vis == 0 or n_llm == 0:
        raise RuntimeError(
            f"LoRA 注入不完整: vision={n_vis}, llm={n_llm} — 检查 target 正则")

    # ⚠️ get_peft_model 会把所有非 adapter 参数重新冻结(v6e-1 烟测发现:
    # freeze_base 打开的 Projector 被静默关掉,Phase 5/5b 将不训 projector)
    # → 此处恢复;KTO 等只训 LoRA 的阶段在调用方自行再冻结
    proj_kw = cfg["freeze"]["projector_keywords"]
    n_proj = 0
    for n, p in peft_model.named_parameters():
        low = n.lower()
        if any(k in low for k in proj_kw) and "lora" not in low:
            p.requires_grad = True
            n_proj += p.numel()
    print(f"[LoRA] injected: llm={n_llm}, vision={n_vis} tensors; "
          f"projector re-enabled: {n_proj/1e6:.1f}M params")
    return peft_model


def build_optimizer(model, cfg, aux_module=None, lr_scale=1.0,
                    ks_module=None):
    """参数分组: LoRA+ (B×16) / vision vs llm / projector / aux heads。"""
    import torch
    lrs = cfg["lr"]
    ratio = cfg["lora"]["loraplus_lr_ratio"]
    # v7 过热哨兵(jax_impl/FINDINGS.md): rsLoRA scale × lr × LoRA+
    # 乘积过大 → 视觉→字母回路训死,loss 曲线无异常。JAX 端实锤,
    # torch 端同款乘法链未单独验证 —— 越线即警告。
    if cfg["lora"].get("use_rslora") and lrs["llm_lora"] * ratio > 1e-4:
        print("⚠️ [optimizer] rsLoRA + lr×LoRA+ 乘积过热"
              f"(llm_lora={lrs['llm_lora']:g} × ratio={ratio:g}): "
              "v7 实测该组合训不进分类字母。已验证冷配方: "
              "llm_lora=2e-5, loraplus_lr_ratio=1;"
              "调参前先过合成数据过拟合门禁")
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
    # 分别校验(真机踩坑: 合并校验时 KS 参数会掩盖 aux 为空的事实)
    if aux_module is not None:
        aux_params = list(aux_module.parameters())
        if not aux_params:
            raise RuntimeError("辅助头参数为空 — eager 初始化未生效,"
                               "参数将不进优化器(随机权重陪跑),拒绝开训")
        groups["aux"] = aux_params
    if ks_module is not None:
        ks_params = list(ks_module.parameters())
        if not ks_params:
            raise RuntimeError("KS 父类头参数为空 — 需以 dim 构造"
                               "(eager),拒绝开训")
        groups["aux"] += ks_params                      # 与 aux 同组同 LR

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
        matched = [mod for name, mod in model.named_modules()
                   if any(k in name.lower() for k in projector_keywords)]
        if not matched:
            raise RuntimeError("projector module not found for aux hook")
        # 钩子挂**顶层** embedder(named_modules 父先于子 → matched[0]):
        # 其输出 = 投影后的最终特征(1536),与下方 2D 权重推断的维度一致。
        # 真机踩坑: 挂最深层匹配会拿到投影前 768 维特征,与头维度错位
        matched[0].register_forward_hook(self._hook)
        # ⚠️ 必须在此(优化器构建前)eager 初始化 —— 懒初始化晚于
        # build_optimizer,辅助头参数将不进优化器(随机权重陪跑,原版 bug)。
        # 维度在**所有**匹配模块中找 2D 权重(真机踩坑: 最深层匹配可能是
        # norm,只有 1D 权重,单看它推不出维度 → eager 静默失败)
        w = next((p for m in matched for p in m.parameters()
                  if p.dim() == 2), None)
        if w is None:
            raise RuntimeError(
                "无法从 projector 模块推断特征维度(未找到 2D 权重)— "
                "eager 初始化失败会让辅助头绕过优化器,拒绝继续")
        self._lazy_init(w.shape[0], "cpu", None)
        print(f"[aux] heads eager-init: dim={w.shape[0]} (fp32)", flush=True)
        return self

    def to(self, device):
        if self.heads is not None:
            self.pool_score.to(device)
            self.heads.to(device)
        return self

    def _hook(self, module, inp, out):
        self._feat = out if not isinstance(out, tuple) else out[0]

    def _lazy_init(self, dim, device, dtype):
        # 头参数固定 fp32(小张量;bf16 + lr1e-4 有更新被舍入吞掉的风险),
        # 特征在 compute_loss 内升 fp32 对齐
        nn = self.nn
        self.pool_score = nn.Linear(dim, 1).to(device)
        self.heads = nn.ModuleDict({
            h: nn.Linear(dim, len(v)).to(device)
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
        f = f.reshape(B, -1, f.shape[-1]).float()   # 帧维并 token 维;fp32 头
        if self.heads is None:               # 兜底(正常已在 attach eager)
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
    """KeyScene 6 大类父类头(Phase 5 轻量辅助)。

    dim 必传(eager 初始化)—— 懒初始化会晚于优化器构建,参数进不了
    优化器(与 AuxHeads 同源 bug,整体 review 修复)。fp32 头。
    """

    def __init__(self, dim=None):
        self.head = None
        if dim:
            import torch.nn as nn
            self.head = nn.Linear(dim, len(KS_CLASSES))

    def to(self, device):
        if self.head is not None:
            self.head.to(device)
        return self

    def compute_loss(self, hidden_states, labels_mask, ks_labels):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        if (ks_labels == -100).all():
            return torch.tensor(0.0, device=hidden_states.device)
        if self.head is None:                 # 兜底
            self.head = nn.Linear(hidden_states.shape[-1],
                                  len(KS_CLASSES)).to(hidden_states.device)
        h = hidden_states.float()             # fp32 头
        m = labels_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        return F.cross_entropy(self.head(pooled), ks_labels,
                               ignore_index=-100)

    def parameters(self):
        return [] if self.head is None else list(self.head.parameters())
