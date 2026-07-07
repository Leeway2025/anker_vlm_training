"""交付物拆分(training_plan 13 节)。

产出两件:
  (a) llm_adapter/   — LLM LoRA adapter(供客户 rkllm-toolkit 转换;
      顺带剥离 vision 的 lora 键 + 可选 issue480 重命名)
  (b) vision_merged.pt — Vision Encoder(LoRA merge 后)+ Projector
      状态字典(供 export_onnx.py → vision_video.rknn 编译源)

辅助头(aux_heads.pt)不进任何交付物 —— 训练期物理分离,天然删除。

用法:
  python -m export.split_deliverables --adapter outputs/swa_final \
      --out deliverables/ [--issue480]
"""
import argparse, os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def split_adapter_keys(state_dict_keys, vision_keywords):
    """纯逻辑可单测: 键分为 llm / vision 两组。"""
    llm, vis = [], []
    for k in state_dict_keys:
        (vis if any(kw in k.lower() for kw in vision_keywords) else llm).append(k)
    return llm, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=None, help="base 模型路径(merge 用)")
    ap.add_argument("--projector", default=None,
                    help="projector.pt(训练后全参权重,vision merge 必传)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--issue480", action="store_true")
    a = ap.parse_args()
    import yaml, torch
    from safetensors.torch import load_file, save_file
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    vis_kw = cfg["freeze"]["vision_keywords"]
    os.makedirs(a.out, exist_ok=True)

    # (a) LLM adapter: 剥离 vision lora 键
    sd = load_file(os.path.join(a.adapter, "adapter_model.safetensors"))
    llm_keys, vis_keys = split_adapter_keys(list(sd), vis_kw)
    llm_sd = {k: sd[k] for k in llm_keys}
    if a.issue480:
        # 仓库相对路径,不依赖 cwd(E2E 实测: 在别的目录运行时 "docs" 失效)
        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "docs"))
        from issue480_workaround import rename_keys
        ren = rename_keys(list(llm_sd))
        llm_sd = {ren.get(k, k): v for k, v in llm_sd.items()}
    llm_dir = os.path.join(a.out, "llm_adapter")
    os.makedirs(llm_dir, exist_ok=True)
    save_file(llm_sd, os.path.join(llm_dir, "adapter_model.safetensors"))
    shutil.copy(os.path.join(a.adapter, "adapter_config.json"), llm_dir)
    print(f"(a) llm_adapter: {len(llm_keys)} keys "
          f"(stripped {len(vis_keys)} vision keys)")

    # (b) vision merge: base + full adapter → merge → 抽 vision/projector
    if a.base:
        from training.common import load_model_and_processor
        from peft import PeftModel
        cfg["model"]["name_or_path"] = a.base
        model, _ = load_model_and_processor(cfg)
        model = PeftModel.from_pretrained(model, a.adapter)
        model = model.merge_and_unload()
        if a.projector:                      # 训练后 projector 覆盖 base 值
            model.load_state_dict(
                torch.load(a.projector, map_location="cpu"), strict=False)
            print(f"loaded trained projector: {a.projector}")
        else:
            print("[WARN] --projector 未传,vision_merged 将含 base 的"
                  " projector(Phase 5 训练成果丢失)")
        keep = vis_kw + cfg["freeze"]["projector_keywords"]
        vis_sd = {k: v.cpu() for k, v in model.state_dict().items()
                  if any(kw in k.lower() for kw in keep)}
        torch.save(vis_sd, os.path.join(a.out, "vision_merged.pt"))
        print(f"(b) vision_merged.pt: {len(vis_sd)} tensors")


if __name__ == "__main__":
    main()
