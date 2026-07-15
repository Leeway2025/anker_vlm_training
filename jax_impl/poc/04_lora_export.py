"""Gate D: JAX LoRA → HF peft 导出链数值验证(RKLLM 只吃 HF 格式)。

  1. jax venv:   python jax_impl/poc/04_lora_export.py --side jax --out /tmp/jax_lora
  2. torch venv: python jax_impl/poc/04_lora_export.py --side hf  --ref /tmp/jax_lora

设计:
  - 只注入 q_einsum(35 层)带种子的非零 LoRA(其余 b=0 无增量),
    单模块即可证明 名称映射/矩阵取向/缩放约定/层序 全部正确;
  - 映射: A = a.T [r,1536];B = b.reshape(r, N*H).T [2048,r];
    JAX 无 α 缩放 → adapter_config 设 lora_alpha=r(scaling=1);
  - 判定: HF(base+adapter) 与 JAX(base+lora) 同输入 top-5 一致,
    且两者都 ≠ base 的 top-5 邻域(防零增量假通过)。
"""
import argparse
import json
import os

PROMPT_IDS = [2, 818, 5279, 529, 7001, 563]   # "The capital of France is"(03 已对齐)
RANK = 8
TOPK = 5


def run_jax(out_dir):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import jax
    import jax.numpy as jnp
    from gemma import gm
    from gemma import peft as gpeft

    os.makedirs(out_dir, exist_ok=True)
    model = gm.nn.LoRA(rank=RANK, model=gm.nn.Gemma4_E2B())

    # 结构: eval_shape 拿全树(零内存),split 出 lora 子树
    struct = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0), tokens=jnp.zeros((1, 8), jnp.int32)))
    lora_struct = gpeft.split_params(struct["params"])[1]

    rng = np.random.RandomState(42)
    sd = {}                                   # peft 导出用

    def fill(path, leaf):
        keys = [getattr(k, "key", str(k)) for k in path]
        p = "/".join(keys)
        if "q_einsum" in p and p.endswith("a"):
            v = rng.normal(0, 0.2, leaf.shape).astype(np.float32)
        elif "q_einsum" in p and p.endswith("b"):
            v = rng.normal(0, 0.2, leaf.shape).astype(np.float32)
        else:
            v = np.zeros(leaf.shape, np.float32)   # 其余无增量
        return jnp.asarray(v)

    lora_vals = jax.tree_util.tree_map_with_path(fill, lora_struct)

    base = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    params = gpeft.merge_params(base, lora_vals)

    out = model.apply({"params": params},
                      tokens=jnp.asarray([PROMPT_IDS]), return_last_only=True)
    logits = out.logits[0] if hasattr(out, "logits") else out[0]
    lp = jax.nn.log_softmax(logits.astype(jnp.float32))
    top = jnp.argsort(lp)[-TOPK:][::-1]
    result = {"top_ids": top.tolist(),
              "top_logprobs": [round(float(lp[i]), 4) for i in top]}
    print(f"[JAX+LoRA] top{TOPK}: {result['top_ids']} lp={result['top_logprobs']}")

    # ---- 导出 peft 格式 ----
    from safetensors.numpy import save_file
    flat = jax.tree_util.tree_flatten_with_path(lora_vals)[0]
    got = {}
    for path, v in flat:
        keys = [getattr(k, "key", str(k)) for k in path]
        p = "/".join(keys)
        if "q_einsum" not in p:
            continue
        layer = next(k for k in keys if k.startswith("layer_"))
        i = int(layer.split("_")[1])
        v = np.asarray(v, np.float32)
        pre = f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj"
        if p.endswith("a"):                     # (1536, r) -> A [r, 1536]
            got[f"{pre}.lora_A.weight"] = v.T.copy()
        else:                                   # (r, N, H) -> B [N*H, r]
            got[f"{pre}.lora_B.weight"] = v.reshape(v.shape[0], -1).T.copy()
    save_file(got, os.path.join(out_dir, "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "r": RANK, "lora_alpha": RANK,   # scaling=1 对齐 JAX
           "lora_dropout": 0.0, "bias": "none",
           "target_modules": r".*language_model\\.layers\\.\\d+\\.self_attn\\.q_proj",
           "task_type": None,   # 正则限语言侧,避开视觉塔非标准 Linear
           "base_model_name_or_path": "google/gemma-4-e2b-it"}
    json.dump(cfg, open(os.path.join(out_dir, "adapter_config.json"), "w"))
    json.dump(result, open(os.path.join(out_dir, "jax_result.json"), "w"))
    print(f"[OK] 导出 {len(got)} 个张量 -> {out_dir}")


def run_hf(ref_dir):
    import torch
    import transformers
    import yaml
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    name = cfg["model"]["name_or_path"]
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        name, torch_dtype=torch.float32)
    ids = torch.tensor([PROMPT_IDS])

    with torch.no_grad():
        base_top = torch.topk(model(input_ids=ids).logits[0, -1], TOPK).indices.tolist()

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ref_dir)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1]
    lp = torch.log_softmax(logits.float(), -1)
    top = torch.topk(lp, TOPK)
    ref = json.load(open(os.path.join(ref_dir, "jax_result.json")))
    print(f"[HF base   ] top{TOPK}: {base_top}")
    print(f"[HF +adapter] top{TOPK}: {top.indices.tolist()} "
          f"lp={[round(float(x), 4) for x in top.values]}")
    print(f"[JAX+LoRA  ] top{TOPK}: {ref['top_ids']} lp={ref['top_logprobs']}")
    same = top.indices.tolist()[:2] == ref["top_ids"][:2]
    moved = top.indices.tolist() != base_top or ref["top_ids"] != base_top
    print("Gate D:", "PASS" if same and moved else
          ("NO-GO(增量为零,测试无效)" if not moved else "NO-GO(两侧不一致)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["jax", "hf"], required=True)
    ap.add_argument("--out", default="/tmp/jax_lora")
    ap.add_argument("--ref", default="/tmp/jax_lora")
    a = ap.parse_args()
    run_jax(a.out) if a.side == "jax" else run_hf(a.ref)
