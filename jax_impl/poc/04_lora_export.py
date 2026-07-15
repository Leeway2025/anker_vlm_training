"""Gate D: JAX LoRA → HF peft 格式导出链验证(RKLLM 只吃 HF 格式)。

  /dev/shm/venv_jax/bin/python jax_impl/poc/04_lora_export.py --out /tmp/jax_lora_hf

步骤:
  1. gm 库注入 LoRA(内省 gm.nn._lora / gm.peft 的真实入口)
  2. 抽出各层 A/B 矩阵,按名称映射表转成 peft 键名
     (语言侧需带 language_model. 前缀 —— RKLLM Issue #480 约定,
      与 export/ 现有导出脚本口径一致)
  3. 存 adapter_model.safetensors + adapter_config.json
  4. 验证: 用 torch venv 的 peft 加载回读,逐张量比对数值

名称映射(JAX param tree → peft key)是本 Gate 的全部难度所在;
先跑通 1 层再批量,映射表沉淀到 jax_impl/export_mapping.py。
"""
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/jax_lora_hf")
    ap.add_argument("--rank", type=int, default=8)   # PoC 用小 rank,链路同构
    a = ap.parse_args()

    from gemma import gm

    # ---- 1. LoRA 注入入口内省 ----
    cands = []
    for mod_name in ("nn", "peft", "ckpts"):
        mod = getattr(gm, mod_name, None)
        if mod:
            cands += [f"gm.{mod_name}.{n}" for n in dir(mod)
                      if "lora" in n.lower()]
    print(f"[introspect] LoRA 入口候选: {cands}")
    if not cands:
        print("[FAIL] gemma 库无 LoRA 入口 —— 检查版本(应有 gm/nn/_lora.py)")
        sys.exit(1)

    # TODO(按内省结果补全):
    #   model = gm.nn.LoRA(rank=a.rank, model=gm.nn.Gemma4_E2B()) 或等价 API
    #   params = <初始化/加载>
    #   lora_tree = <从 params 过滤出 lora_a/lora_b>
    #   for jax_name, (A, B) in lora_tree:
    #       peft_key = map_name(jax_name)   # jax_impl/export_mapping.py
    #       sd[f"base_model.model.{peft_key}.lora_A.weight"] = np(A).T ...
    #   safetensors.numpy.save_file(sd, out/adapter_model.safetensors)
    print("\n[NEXT] 按上面候选补全注入段;映射表放 jax_impl/export_mapping.py")
    print("Gate D: PENDING(注入/映射段待按真实 API 补全)")


if __name__ == "__main__":
    main()
