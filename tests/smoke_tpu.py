"""TPU 真机一键烟测(v6e-8;单进程即可,不需 8 核分布式)。

用法(在 torch_xla venv 里,repo 根目录):
    PJRT_DEVICE=TPU python tests/smoke_tpu.py [--model google/gemma-4-e2b-it]

按序执行 8 项检查,任何一项 FAIL 都打印诊断和修复指引;
全部 PASS 后即可启动正式训练(docs/TPU_SETUP.md)。
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = []


def check(name):
    def deco(fn):
        def wrapper(ctx):
            t0 = time.time()
            try:
                fn(ctx)
                RESULTS.append((name, "PASS", f"{time.time()-t0:.1f}s"))
                print(f"  ✅ {name} ({time.time()-t0:.1f}s)")
            except Exception as e:
                RESULTS.append((name, "FAIL", str(e)[:200]))
                print(f"  ❌ {name}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
            return ctx
        return wrapper
    return deco


@check("1. torch_xla 设备可见")
def s1_devices(ctx):
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    n = len(xm.get_xla_supported_devices())
    print(f"     device={dev}, visible={n}")
    assert n >= 1, "无 TPU 设备 — 确认 PJRT_DEVICE=TPU 且无 JAX 进程占用"
    ctx["device"] = dev


@check("2. 模型 + processor 加载")
def s2_load(ctx):
    import yaml
    from training.common import load_model_and_processor
    cfg = yaml.safe_load(open("configs/base.yaml"))
    cfg["model"]["name_or_path"] = ctx["model_name"]
    model, processor = load_model_and_processor(cfg)
    print(f"     class={type(model).__name__}, "
          f"processor={type(processor).__name__}")
    ctx.update(cfg=cfg, model=model, processor=processor)


@check("3. processor 视频入参签名(2026-07-07 已确认,此处回归验证)")
def s3_processor_sig(ctx):
    import numpy as np
    p = ctx["processor"]
    frames = np.zeros((16, 384, 384, 3), dtype=np.uint8)   # 均匀 16 帧
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames},
        {"type": "text", "text": "describe"}]}]
    out = p.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", do_sample_frames=False)
    keys = list(out.keys())
    print(f"     输出键: {keys}")
    assert "pixel_values_videos" in keys, f"视频张量键缺失: {keys}"
    pv = out["pixel_values_videos"]
    assert pv.shape[1] == 16, \
        f"帧数被改写: {pv.shape}(do_sample_frames=False 未生效?)"
    # 70 token/帧 × 16 帧 = 1120 视觉 token(方案预算),序列应 > 1120
    assert out["input_ids"].shape[1] > 1120, \
        f"prompt 序列过短 {out['input_ids'].shape},视觉 token 未展开?"
    print(f"     pixel_values_videos {tuple(pv.shape)}, "
          f"prompt len {out['input_ids'].shape[1]} ✓")
    ctx["pixel_key"] = "pixel_values_videos"


@check("4. freeze_base(PLE/embed 命名验证 — 多 LoRA 红线)")
def s4_freeze(ctx):
    from training.common import freeze_base
    stats = freeze_base(ctx["model"], ctx["cfg"])
    print(f"     {stats['frozen_keyword_hits']}")
    print(f"     projector trainable: "
          f"{stats['projector_trainable_params']/1e6:.1f}M params")
    hits = stats["frozen_keyword_hits"]
    if not any(hits.get(k, 0) for k in
               ("per_layer", "ple", "per_layer_embedding")):
        # 打印疑似 PLE 参数名帮助定位
        cand = [n for n, _ in ctx["model"].named_parameters()
                if "layer" in n.lower() and "embed" in n.lower()][:5]
        raise RuntimeError(
            f"PLE 关键字零命中!疑似候选: {cand} — "
            f"更新 configs/base.yaml freeze.keywords")


@check("5. layer_types → global 层检测(差异化 rank 依据)")
def s5_layers(ctx):
    from training.common import detect_global_layers
    glb = detect_global_layers(ctx["model"])
    print(f"     global layers: {glb}")
    assert len(glb) >= 3, f"global 层过少({glb}),检查 layer_types 解析"
    ctx["global_layers"] = glb


@check("6. LoRA 注入(rank_pattern + rsLoRA + PISSA)")
def s6_lora(ctx):
    from training.common import build_lora
    model = build_lora(ctx["model"], ctx["cfg"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"     trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B "
          f"({100*trainable/total:.2f}%)")
    # 验证 global 层确实拿到 r=512
    r_glb = ctx["cfg"]["lora"]["r_global"]
    # 限定 language_model —— vision_tower 也有 layers.{i},其 r=256 是正确的
    hit = [n for n, p in model.named_parameters()
           if "language_model" in n
           and f"layers.{ctx['global_layers'][0]}." in n
           and "lora_a" in n.lower()]
    if hit:
        shape = dict(model.named_parameters())[hit[0]].shape
        print(f"     global 层 lora_A 形状: {shape}(应含 {r_glb})")
        assert r_glb in shape, f"rank_pattern 未生效: {shape}"
    # get_peft_model 会重新冻结非 adapter 参数,build_lora 须恢复 projector
    proj_kw = ctx["cfg"]["freeze"]["projector_keywords"]
    n_proj = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and "lora" not in n.lower()
                 and any(k in n.lower() for k in proj_kw))
    assert n_proj > 0, "projector 在 LoRA 注入后被冻结(应恢复全参训练)"
    print(f"     projector 仍可训: {n_proj/1e6:.1f}M ✓")
    ctx["model"] = model


@check("7. 前向+反向一步(bf16, XLA 编译)")
def s7_forward(ctx):
    import torch
    model = ctx["model"].to(ctx["device"])
    L = 512                                     # 烟测用短序列
    ids = torch.randint(5, 1000, (2, L)).to(ctx["device"])
    labels = ids.clone()
    labels[:, :L // 2] = -100
    t0 = time.time()
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                labels=labels)
    out.loss.backward()
    import torch_xla.core.xla_model as xm
    xm.mark_step()
    print(f"     loss={out.loss.item():.3f}, 首步(含编译)={time.time()-t0:.0f}s")
    # 第二步同 shape 应显著变快(编译缓存命中)
    t1 = time.time()
    out2 = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                 labels=labels)
    out2.loss.backward()
    xm.mark_step()
    dt2 = time.time() - t1
    print(f"     次步={dt2:.1f}s(应远小于首步 → 无重编译)")


@check("8. collator 端到端(固定 padding + 分类 token 加权)")
def s8_collator(ctx):
    import numpy as np
    from data.build_dataset import AnkerCollator
    from data.formatting import build_target
    coll = AnkerCollator(ctx["processor"], ctx["cfg"])
    spec = build_target("B", "i", "A delivery person places a package.")
    ex = {"frames": np.zeros((16, 384, 384, 3), np.uint8),
          "target_spec": spec, "prompt_suffix": "",
          "aux_labels": {}, "ks_label": -100, "rt": "B", "sk": "i",
          "video_id": "smoke"}
    batch = coll([ex, ex])
    L = ctx["cfg"]["train"]["max_seq_len"]
    assert batch["input_ids"].shape[1] == L, \
        f"固定 padding 未生效: {batch['input_ids'].shape} != {L}"
    w = batch["token_weights"]
    assert (w == 4.0).any() and (w == 1.0).any(), "分类加权丢失"
    assert batch["pixel_values_videos"].shape[:2] == (2, 16), \
        f"视频张量形状异常: {batch['pixel_values_videos'].shape}"
    print(f"     input_ids {tuple(batch['input_ids'].shape)}, "
          f"pixel_values_videos {tuple(batch['pixel_values_videos'].shape)}, "
          f"weights 含 4.0/1.0 ✓")
    ctx["batch"] = batch


@check("9. 视频前向+反向端到端(collator batch → TPU,加权 CE)")
def s9_video_forward(ctx):
    import torch
    import torch_xla.core.xla_model as xm
    model, d = ctx["model"], ctx["device"]
    # 与真实训练配置一致: XLA 专用 gradient checkpointing 必开
    # (原生 checkpoint 被 CSE 优化掉,v6e-1 实测无效 → 35.4G OOM)
    from training.common import enable_xla_gradient_checkpointing
    enable_xla_gradient_checkpointing(model)
    model.train()   # checkpoint 分支仅在 training 模式生效(v6e-1 踩坑:
                    # eval 模式下 enable 了也不进图,显存字节不差)
    batch = {k: v.to(d) for k, v in ctx["batch"].items()}
    weights = batch.pop("token_weights")
    batch.pop("ks_labels"), batch.pop("aux_labels")
    labels = batch.pop("labels")
    t0 = time.time()
    out = model(**batch)
    s_logits = out.logits[:, :-1]
    s_labels, s_w = labels[:, 1:], weights[:, 1:]
    import torch.nn.functional as F
    ce = F.cross_entropy(
        s_logits.float().reshape(-1, s_logits.size(-1)),
        s_labels.reshape(-1), reduction="none",
        ignore_index=-100).view(s_labels.shape)
    valid = (s_labels != -100).float()
    loss = (ce * s_w * valid).sum() / (s_w * valid).sum().clamp_min(1.0)
    loss.backward()
    xm.mark_step()
    lv = float(loss)
    print(f"     加权 CE loss={lv:.3f}, 首步(含编译)={time.time()-t0:.0f}s")
    assert lv == lv and lv < 100, f"loss 异常: {lv}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-e2b-it")
    a = ap.parse_args()
    ctx = {"model_name": a.model}
    print(f"=== TPU 烟测: {a.model} ===")
    for fn in [s1_devices, s2_load, s3_processor_sig, s4_freeze,
               s5_layers, s6_lora, s7_forward, s8_collator,
               s9_video_forward]:
        ctx = fn(ctx)
        if RESULTS[-1][1] == "FAIL" and RESULTS[-1][0][0] in "1234":
            print("\n⛔ 前置步骤失败,后续跳过")
            break
    print("\n=== 汇总 ===")
    for name, st, info in RESULTS:
        print(f"  [{st}] {name}  {info}")
    n_fail = sum(1 for _, st, _ in RESULTS if st == "FAIL")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
