"""RKLLM Issue #480 workaround(框架无关,客户侧编译前使用)。

现象: rkllm-toolkit 对 VLM 的 LoRA 报
  "Found unsupported layer: model.language_model.layers.X...lora_A.weight"
原因: VLM 权重路径带 language_model. 前缀,toolkit 只识别 model.* 。
本脚本重命名 adapter safetensors 的键: .language_model. → .

用法: python issue480_workaround.py --adapter adapter_model.safetensors --out fixed.safetensors
"""
import argparse


def rename_keys(keys):
    """纯逻辑,可单测: 返回 {old: new},仅含需要改名的键。"""
    out = {}
    for k in keys:
        if ".language_model." in k:
            out[k] = k.replace(".language_model.", ".")
        elif k.startswith("language_model."):
            out[k] = k[len("language_model."):]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors.torch import load_file, save_file
    sd = load_file(a.adapter)
    ren = rename_keys(list(sd))
    new = {ren.get(k, k): v for k, v in sd.items()}
    assert len(new) == len(sd), "key collision after rename!"
    save_file(new, a.out)
    print(f"renamed {len(ren)}/{len(sd)} keys -> {a.out}")


if __name__ == "__main__":
    main()
