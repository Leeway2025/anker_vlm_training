"""JAX LoRA → HF peft 完整导出(LLM 全模块)+ 自检对拍。

  导出:  python jax_impl/export_hf.py --npz out/train_params.npz --out DIR
  自检:  python jax_impl/export_hf.py --selftest --out /tmp/lora_full --side jax
         <torch venv> python jax_impl/export_hf.py --selftest --out ... --side hf

映射表(Gate D 已证 q_proj 链;其余同构展开):
  q_einsum      a(1536,r) b(r,N,H)   → q_proj  A=aᵀ, B=b.reshape(r,-1)ᵀ
  kv_einsum     a(1536,r) b(r,2,K,H) → k/v_proj A=aᵀ(共享), B=b[:,i].reshape(r,-1)ᵀ
  attn_vec      a(N,H,r)  b(r,1536)  → o_proj  A=a.reshape(-1,r)ᵀ, B=bᵀ
  gating_einsum a(1536,r) b(r,2,F)   → gate/up  A=aᵀ(共享), B=b[:,i]ᵀ
  mlp/linear    a(F,r)    b(r,1536)  → down_proj A=aᵀ, B=bᵀ
缩放: JAX 无 α → adapter_config lora_alpha=r(scaling=1)。
视觉塔 LoRA 导出与 projector 的 HF 键名映射见 FINDINGS(TODO v3)。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT_IDS = [2, 818, 5279, 529, 7001, 563]
TOPK = 5


def map_llm_loras(flat):
    """flat: {jax_path_str: np.ndarray}(仅 lora 叶)→ {peft_key: np.ndarray}"""
    import numpy as np
    out = {}
    pre = "base_model.model.model.language_model.layers"
    by_mod = {}
    skipped = []          # 有映射盲区必须出声(视觉塔/embedder LoRA)
    for p, v in flat.items():
        if not (p.startswith("lora/") or p.startswith("layer_")
                or "/layer_" in p):
            if p.startswith("lora/"):
                skipped.append(p)
            continue
        parts = p.split("/")
        li = next((x for x in parts if x.startswith("layer_")), None)
        if li is None:
            skipped.append(p)
            continue
        i = int(li.split("_")[1])
        kind = ("q" if "q_einsum" in p else
                "kv" if "kv_einsum" in p else
                "o" if "attn_vec_einsum" in p else
                "gate_up" if "gating_einsum" in p else
                "down" if "/mlp/" in p else None)
        if kind is None:
            skipped.append(p)
            continue
        ab = "a" if p.endswith("/a") else "b"
        by_mod.setdefault((i, kind), {})[ab] = np.asarray(v, np.float32)

    if skipped:
        vis = sum(1 for k in skipped if "vision_encoder" in k)
        print(f"⚠️ [export] {len(skipped)} 个 LoRA 键无 HF 映射、未导出"
              f"(vision_encoder {vis} 个)。权重仍完整保留在源 npz,"
              f"不丢失;JAX 侧推理不受影响,仅 torch 交付缺视觉适配。"
              f"示例: {skipped[0]}")

    for (i, kind), d in sorted(by_mod.items()):
        a, b = d.get("a"), d.get("b")
        if a is None or b is None:
            continue
        L = f"{pre}.{i}"
        if kind == "q":
            out[f"{L}.self_attn.q_proj.lora_A.weight"] = a.T.copy()
            out[f"{L}.self_attn.q_proj.lora_B.weight"] = \
                b.reshape(b.shape[0], -1).T.copy()
        elif kind == "kv":
            for j, name in enumerate(("k_proj", "v_proj")):
                out[f"{L}.self_attn.{name}.lora_A.weight"] = a.T.copy()
                out[f"{L}.self_attn.{name}.lora_B.weight"] = \
                    b[:, j].reshape(b.shape[0], -1).T.copy()
        elif kind == "o":
            out[f"{L}.self_attn.o_proj.lora_A.weight"] = \
                a.reshape(-1, a.shape[-1]).T.copy()
            out[f"{L}.self_attn.o_proj.lora_B.weight"] = b.T.copy()
        elif kind == "gate_up":
            for j, name in enumerate(("gate_proj", "up_proj")):
                out[f"{L}.mlp.{name}.lora_A.weight"] = a.T.copy()
                out[f"{L}.mlp.{name}.lora_B.weight"] = b[:, j].T.copy()
        elif kind == "down":
            out[f"{L}.mlp.down_proj.lora_A.weight"] = a.T.copy()
            out[f"{L}.mlp.down_proj.lora_B.weight"] = b.T.copy()
    return out


TARGET_RE = (r".*language_model\.layers\.\d+\."
             r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")


def write_adapter(sd, rank, out_dir):
    from safetensors.numpy import save_file
    os.makedirs(out_dir, exist_ok=True)
    save_file(sd, os.path.join(out_dir, "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "r": rank, "lora_alpha": rank,
               "lora_dropout": 0.0, "bias": "none",
               "target_modules": TARGET_RE, "task_type": None,
               "base_model_name_or_path": "google/gemma-4-e2b-it"},
              open(os.path.join(out_dir, "adapter_config.json"), "w"))
    print(f"[export] {len(sd)} tensors -> {out_dir}")


def run_export(npz_path, out_dir, scheme="uniform"):
    import numpy as np
    z = np.load(npz_path)
    flat = {k.removeprefix("lora/"): z[k] for k in z.files
            if "/lora/" in k or k.startswith("lora/")}
    sd = map_llm_loras(flat)
    if scheme == "prod":
        # prod 前向已带 peft 同款缩放 → 张量原样导出,配置声明
        # rank/alpha_pattern + rsLoRA(与 torch 生产 adapter 同款)
        from jax_impl.prod_lora import prod_adapter_config
        import json as _json
        os.makedirs(out_dir, exist_ok=True)
        from safetensors.numpy import save_file
        save_file(sd, os.path.join(out_dir, "adapter_model.safetensors"))
        _json.dump(prod_adapter_config(),
                   open(os.path.join(out_dir, "adapter_config.json"), "w"))
        print(f"[export/prod] {len(sd)} tensors -> {out_dir}")
        proj = {k: z[k] for k in z.files if "mm_input_projection" in k}
        if proj:
            np.savez(os.path.join(out_dir, "projector_params.npz"), **proj)
        return
    rank = next(v.shape[0] for k, v in sd.items() if k.endswith("lora_A.weight"))
    write_adapter(sd, rank, out_dir)
    proj = {k: z[k] for k in z.files if "mm_input_projection" in k}
    if proj:
        np.savez(os.path.join(out_dir, "projector_params.npz"), **proj)
        print(f"[export] projector -> projector_params.npz "
              f"(HF 侧加载映射见 FINDINGS TODO)")


def run_selftest_jax(out_dir, rank=8, scheme="uniform"):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import jax
    import jax.numpy as jnp
    from gemma import gm
    from gemma import peft as gpeft

    if scheme == "prod":
        from jax_impl.prod_lora import install_prod_lora
        install_prod_lora()
        rank = 256                     # 占位;真实 rank 由 prod patch 按层定
    model = gm.nn.LoRA(rank=rank, model=gm.nn.Gemma4_E2B())
    struct = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0), tokens=jnp.zeros((1, 8), jnp.int32)))
    lora_struct = gpeft.split_params(struct["params"])[1]
    rng = np.random.RandomState(7)

    only = os.environ.get("ONLY", "")     # 逐模块消融: kv/o/gating/linear…
    # prod 方案前向自带 32~45× 缩放,行为对拍的扰动须等比缩小,
    # 否则模型被推进近平局混沌区,实现级噪声即可重排 top-5(踩坑 ×3)
    std = 0.0003 if scheme == "prod" else 0.02   # ΔW≪权重,保持线性区
    def fill(path, leaf):
        p = "/".join(getattr(k, "key", str(k)) for k in path)
        llm = p.startswith("layer_") or "/layer_" in p
        if only and only not in p:
            llm = False
        return jnp.asarray(rng.normal(0, std, leaf.shape), jnp.float32) \
            if llm else jnp.zeros(leaf.shape, jnp.float32)
    lora_vals = jax.tree_util.tree_map_with_path(fill, lora_struct)

    base = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    # 自检必须 fp32: bf16 base 的精度差在大扇面(6144)模块的随机扰动下
    # 会被混沌放大成完全不同的 top-5(权重级对照证明映射本身正确)
    base = jax.tree.map(lambda x: x.astype(jnp.float32), base)
    params = gpeft.merge_params(base, lora_vals)
    out = model.apply({"params": params},
                      tokens=jnp.asarray([PROMPT_IDS]), return_last_only=True)
    logits = out.logits[0] if hasattr(out, "logits") else out[0]
    lp = jax.nn.log_softmax(logits.astype(jnp.float32))
    top = jnp.argsort(lp)[-TOPK:][::-1]
    res = {"top_ids": top.tolist(),
           "top_logprobs": [round(float(lp[i]), 4) for i in top]}
    print(f"[JAX all-module LoRA] top{TOPK}: {res}")

    flat = {"/".join(getattr(k, "key", str(k)) for k in p): np.asarray(v)
            for p, v in jax.tree_util.tree_flatten_with_path(lora_vals)[0]}
    sd = map_llm_loras(flat)
    if scheme == "prod":
        from jax_impl.prod_lora import prod_adapter_config, E2B_GLOBAL_LAYERS
        from safetensors.numpy import save_file
        os.makedirs(out_dir, exist_ok=True)
        save_file(sd, os.path.join(out_dir, "adapter_model.safetensors"))
        json.dump(prod_adapter_config(),
                  open(os.path.join(out_dir, "adapter_config.json"), "w"))
        print(f"[export/prod] {len(sd)} tensors -> {out_dir}")
        # ---- 跨侧矩阵级对拍(确定性,免疫混沌):
        #      JAX 原始 a/b 按 JAX 前向公式 vs 导出 A/B 按 peft 公式 ----
        import math
        pre = "base_model.model.model.language_model.layers"
        rng2 = np.random.RandomState(9)
        worst = 0.0
        for i in (0, 4):                      # 滑动层 + 全局层
            r = 512 if i in E2B_GLOBAL_LAYERS else 256
            s_j = (2 * r) / math.sqrt(r)      # JAX 前向缩放(prod patch)
            x = rng2.normal(0, 1, 1536).astype(np.float32)
            # q: jax einsum '...F,Fr,rNH'
            a = np.asarray(flat[f"layer_{i}/attn/q_einsum/_LoRAEinsum_0/lora/a"])
            b = np.asarray(flat[f"layer_{i}/attn/q_einsum/_LoRAEinsum_0/lora/b"])
            yj = s_j * ((x @ a) @ b.reshape(b.shape[0], -1))
            A = sd[f"{pre}.{i}.self_attn.q_proj.lora_A.weight"]
            B = sd[f"{pre}.{i}.self_attn.q_proj.lora_B.weight"]
            yh = s_j * (B @ (A @ x))          # peft 同层同公式(config 已核)
            worst = max(worst, float(np.abs(yj - yh).max()
                                     / max(np.abs(yj).max(), 1e-9)))
            # kv → k
            a = np.asarray(flat[f"layer_{i}/attn/kv_einsum/_LoRAEinsum_0/lora/a"])
            b = np.asarray(flat[f"layer_{i}/attn/kv_einsum/_LoRAEinsum_0/lora/b"])
            yj = s_j * np.einsum("r,rkh->kh", x @ a, b[:, 0])[0]
            A = sd[f"{pre}.{i}.self_attn.k_proj.lora_A.weight"]
            B = sd[f"{pre}.{i}.self_attn.k_proj.lora_B.weight"]
            yh = s_j * (B @ (A @ x))
            worst = max(worst, float(np.abs(yj - yh).max()
                                     / max(np.abs(yj).max(), 1e-9)))
            # gating → gate
            a = np.asarray(flat[f"layer_{i}/mlp/_LoRAEinsum_gating_einsum/lora/a"])
            b = np.asarray(flat[f"layer_{i}/mlp/_LoRAEinsum_gating_einsum/lora/b"])
            yj = s_j * np.einsum("r,rh->h", x @ a, b[:, 0])
            A = sd[f"{pre}.{i}.mlp.gate_proj.lora_A.weight"]
            B = sd[f"{pre}.{i}.mlp.gate_proj.lora_B.weight"]
            yh = s_j * (B @ (A @ x))
            worst = max(worst, float(np.abs(yj - yh).max()
                                     / max(np.abs(yj).max(), 1e-9)))
        print(f"[matrix-check] 跨侧相对误差 worst={worst:.2e} "
              f"{'PASS' if worst < 1e-5 else 'NO-GO'}")
    else:
        write_adapter(sd, rank, out_dir)
    json.dump(res, open(os.path.join(out_dir, "jax_result.json"), "w"))


def run_selftest_hf(out_dir):
    import torch
    import transformers
    import yaml
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        cfg["model"]["name_or_path"], torch_dtype=torch.float32)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, out_dir)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([PROMPT_IDS])).logits[0, -1]
    lp = torch.log_softmax(logits.float(), -1)
    top = torch.topk(lp, TOPK)
    ref = json.load(open(os.path.join(out_dir, "jax_result.json")))
    print(f"[HF +adapter] top{TOPK}: {top.indices.tolist()} "
          f"lp={[round(float(x), 4) for x in top.values]}")
    print(f"[JAX        ] top{TOPK}: {ref['top_ids']} lp={ref['top_logprobs']}")
    same = top.indices.tolist()[:2] == ref["top_ids"][:2]
    d1 = abs(float(top.values[0]) - ref["top_logprobs"][0])
    print("全模块导出对拍:", "PASS" if same and d1 < 0.15 else "NO-GO")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--side", choices=["jax", "hf"], default="jax")
    ap.add_argument("--scheme", choices=["uniform", "prod"], default="uniform")
    a = ap.parse_args()
    if a.selftest:
        if a.side == "jax":
            run_selftest_jax(a.out, scheme=a.scheme)
        else:
            run_selftest_hf(a.out)
    else:
        run_export(a.npz, a.out, scheme=a.scheme)
