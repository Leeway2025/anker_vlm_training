"""生成 tiny LoRA 测试权重(分工表我方交付物: 客户用它测 RKLLM 转换
链路与 Issue #480,不含任何训练成果)。

产出:
  outputs/tiny_lora/standard/   r=4 纯标准 LoRA(LLM 侧全部 *_proj 键)
  outputs/tiny_lora/issue480/   同上但键名过 issue480 重命名
用法: python scripts/make_tiny_lora.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import yaml
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from training.common import load_model_and_processor

    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    model, _ = load_model_and_processor(cfg)
    lcfg = LoraConfig(                       # 纯标准 LoRA(端侧兼容口径)
        r=4, lora_alpha=8, lora_dropout=0.0, use_rslora=False,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|"
                       r"gate_proj|up_proj|down_proj)",
        task_type="CAUSAL_LM")
    pm = get_peft_model(model, lcfg)
    from peft import get_peft_model_state_dict
    sd = {k: v.detach().cpu().contiguous().to(torch.float16)
          for k, v in get_peft_model_state_dict(pm).items()}

    out = "outputs/tiny_lora"
    for variant in ("standard", "issue480"):
        d = os.path.join(out, variant)
        os.makedirs(d, exist_ok=True)
        cur = dict(sd)
        if variant == "issue480":
            sys.path.insert(0, os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), "docs"))
            from issue480_workaround import rename_keys
            ren = rename_keys(list(cur))
            cur = {ren.get(k, k): v for k, v in cur.items()}
        save_file(cur, os.path.join(d, "adapter_model.safetensors"))
        pm.peft_config["default"].save_pretrained(d)
        print(f"{d}: {len(cur)} tensors, "
              f"{sum(v.numel() for v in cur.values())/1e6:.1f}M params")


if __name__ == "__main__":
    main()
