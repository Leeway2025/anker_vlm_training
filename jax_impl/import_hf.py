"""torch/HF peft adapter → JAX lora npz(export_hf 的逆;跨框架续训入口)。

  python jax_impl/import_hf.py --adapter <dir 含 adapter_model.safetensors> \
      --out imported.npz [--selftest]

结构差与解法(FINDINGS「跨框架续训」):
  - q/o/down: 逐张量精确可逆;
  - k/v 与 gate/up: JAX 融合算子共享 lora A,torch 两侧 A 独立 →
    rank 加倍拼接精确表示: a=[A_kᵀ|A_vᵀ] (in,2r),b 分块放置交叉置零。
  - 全局 rank = 2×r_torch(q/o/down 零填充到 2r)。
  - 续训后再导出 rank 为 2r,RKLLM 预算需确认;可 SVD 收缩(未含)。
--selftest: 导入后 JAX 前向 vs HF+adapter 前向对拍(文本 prompt,
  fp32;vision lora 键跳过 —— 文本前向不受其影响)。
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT_IDS = [2, 818, 5279, 529, 7001, 563]
TOPK = 5


def import_adapter(adapter_dir):
    import numpy as np
    from safetensors import safe_open
    f = safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), "np")
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    alpha0 = float(cfg.get("lora_alpha", cfg["r"]))
    rslora = bool(cfg.get("use_rslora", False))
    # torch 路线是差异化 rank(全局层 512/滑动层 256)→ 逐张量取实际 r,
    # 全局容器 R = 2×max_r,统一零填充

    ten = {}
    pat = re.compile(r"language_model\.layers\.(\d+)\.(self_attn|mlp)"
                     r"\.(\w+_proj)\.lora_(A|B)\.weight$")
    for k in f.keys():
        m = pat.search(k)
        if m:
            i, _, proj, ab = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            ten[(i, proj, ab)] = f.get_tensor(k).astype(np.float32)

    import math
    r_of = {k: v.shape[0] for k, v in ten.items() if k[2] == "A"}
    R = 2 * max(r_of.values())
    print(f"[import] per-module ranks={sorted(set(r_of.values()))} 容器 R={R}")
    print(f"[import] alpha0={alpha0} rslora={rslora} "
          f"示例 scale: r256→{alpha0/(math.sqrt(256) if rslora else 256):.3f} "
          f"r512→{alpha0/(math.sqrt(512) if rslora else 512):.3f}")

    alpha_pat = cfg.get("alpha_pattern") or {}

    def sc(key):                          # 每模块缩放(alpha/r 或 rsLoRA)
        i, proj = key[0], key[1]
        r = r_of[(i, proj, "A")]
        mod = "self_attn" if proj.split("_")[0] in "qkvo" else "mlp"
        # 匹配串须带前导段与模块段,否则 peft 的 .*\.layers\.N\..*\.x_proj
        # 形式正则永不命中(踩坑: 全局层 alpha=1024 被漏,全用了 512)
        name = f"m.layers.{i}.{mod}.{proj}"
        alpha = alpha0
        for pat, v in alpha_pat.items():
            if re.search(pat, name):
                alpha = float(v)
        return alpha / (math.sqrt(r) if rslora else r)

    layers = sorted({i for i, _, _ in ten})
    out = {}

    def put(i, mod, a_arr, b_arr):
        out[f"lora/layer_{i}/{mod}/lora/a"] = a_arr
        out[f"lora/layer_{i}/{mod}/lora/b"] = b_arr

    for i in layers:
        g = lambda p, ab: ten.get((i, p, ab))
        # q: A[r,1536] B[NH,r] → a (1536,2r) 零填, b (2r,N,H)
        A, B = g("q_proj", "A"), g("q_proj", "B")
        if A is not None:
            r = A.shape[0]; scale = sc((i, "q_proj", "A"))
            NH = B.shape[0]
            a = np.zeros((A.shape[1], R), np.float32); a[:, :r] = A.T
            b = np.zeros((R, NH), np.float32); b[:r] = (scale * B).T
            put(i, "attn/q_einsum/_LoRAEinsum_0",
                a, b.reshape(R, 8, NH // 8))
        # kv: 共享 a → rank 拼接
        Ak, Bk = g("k_proj", "A"), g("k_proj", "B")
        Av, Bv = g("v_proj", "A"), g("v_proj", "B")
        if Ak is not None:
            r = Ak.shape[0]; scale = sc((i, "k_proj", "A"))
            H = Bk.shape[0]
            a = np.zeros((Ak.shape[1], R), np.float32)
            a[:, :r] = Ak.T; a[:, r:2 * r] = Av.T
            b = np.zeros((R, 2, 1, H), np.float32)
            b[:r, 0, 0] = (scale * Bk).T
            b[r:2 * r, 1, 0] = (scale * Bv).T
            put(i, "attn/kv_einsum/_LoRAEinsum_0", a, b)
        # o: A[r,NH] B[1536,r]
        A, B = g("o_proj", "A"), g("o_proj", "B")
        if A is not None:
            r = A.shape[0]; scale = sc((i, "o_proj", "A"))
            NH = A.shape[1]
            a = np.zeros((8, NH // 8, R), np.float32)
            a[..., :r] = A.T.reshape(8, NH // 8, r)
            b = np.zeros((R, B.shape[0]), np.float32)
            b[:r] = (scale * B).T
            put(i, "attn/attn_vec_einsum/_LoRAEinsum_0", a, b)
        # gate/up: 共享 a → rank 拼接
        Ag, Bg = g("gate_proj", "A"), g("gate_proj", "B")
        Au, Bu = g("up_proj", "A"), g("up_proj", "B")
        if Ag is not None:
            r = Ag.shape[0]; scale = sc((i, "gate_proj", "A"))
            Hd = Bg.shape[0]
            a = np.zeros((Ag.shape[1], R), np.float32)
            a[:, :r] = Ag.T; a[:, r:2 * r] = Au.T
            b = np.zeros((R, 2, Hd), np.float32)
            b[:r, 0] = (scale * Bg).T
            b[r:2 * r, 1] = (scale * Bu).T
            put(i, "mlp/_LoRAEinsum_gating_einsum", a, b)
        # down
        A, B = g("down_proj", "A"), g("down_proj", "B")
        if A is not None:
            r = A.shape[0]; scale = sc((i, "down_proj", "A"))
            a = np.zeros((A.shape[1], R), np.float32); a[:, :r] = A.T
            b = np.zeros((R, B.shape[0]), np.float32)
            b[:r] = (scale * B).T
            put(i, "mlp/_LoRAEinsum_linear", a, b)
    return out, R


def selftest(npz_path, rank, adapter_dir, side):
    if side == "jax":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import numpy as np
        import jax
        import jax.numpy as jnp
        from gemma import gm
        from gemma import peft as gpeft
        model = gm.nn.LoRA(rank=rank, model=gm.nn.Gemma4_E2B())
        z = np.load(npz_path)
        struct = jax.eval_shape(lambda: model.init(
            jax.random.PRNGKey(0), tokens=jnp.zeros((1, 8), jnp.int32)))
        lora_struct = gpeft.split_params(struct["params"])[1]

        def fill(path, leaf):
            k = "lora/" + "/".join(getattr(x, "key", str(x)) for x in path)
            if k in z.files:
                assert z[k].shape == leaf.shape, f"{k}: {z[k].shape}≠{leaf.shape}"
                return jnp.asarray(z[k], jnp.float32)
            return jnp.zeros(leaf.shape, jnp.float32)
        lora = jax.tree_util.tree_map_with_path(fill, lora_struct)
        base = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
        base = jax.tree.map(lambda x: x.astype(jnp.float32), base)
        params = gpeft.merge_params(base, lora)
        out = model.apply({"params": params},
                          tokens=jnp.asarray([PROMPT_IDS]),
                          return_last_only=True)
        lg = out.logits[0] if hasattr(out, "logits") else out[0]
        lp = jax.nn.log_softmax(lg.astype(jnp.float32))
        top = jnp.argsort(lp)[-TOPK:][::-1]
        r = {"top_ids": top.tolist(),
             "top_logprobs": [round(float(lp[i]), 4) for i in top]}
        print(f"[JAX imported] {r}")
        json.dump(r, open("/tmp/import_jax.json", "w"))
    else:
        import torch
        import transformers
        import yaml
        cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            cfg["model"]["name_or_path"], torch_dtype=torch.float32)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        try:   # 校准: peft 实际 per-module scaling(与 import 的 sc() 对照)
            for n, m in model.named_modules():
                if n.endswith("layers.0.self_attn.q_proj") or \
                        n.endswith("layers.4.self_attn.q_proj"):
                    print(f"[peft-scaling] {n.split('model.')[-1]}: "
                          f"r={m.r} alpha={m.lora_alpha} "
                          f"scaling={m.scaling}")
        except Exception as e:  # noqa: BLE001
            print(f"[peft-scaling] introspect failed: {e}")
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([PROMPT_IDS])).logits[0, -1]
        lp = torch.log_softmax(logits.float(), -1)
        top = torch.topk(lp, TOPK)
        ref = json.load(open("/tmp/import_jax.json"))
        print(f"[HF +adapter ] top: {top.indices.tolist()} "
              f"lp={[round(float(x), 4) for x in top.values]}")
        print(f"[JAX imported] top: {ref['top_ids']} lp={ref['top_logprobs']}")
        same = top.indices.tolist()[:2] == ref["top_ids"][:2]
        print("import 对拍:", "PASS" if same else "NO-GO")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="/tmp/imported_lora.npz")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--side", choices=["jax", "hf"], default="jax")
    a = ap.parse_args()
    if a.selftest and a.side == "hf":
        selftest(a.out, 0, a.adapter, "hf")
    else:
        import numpy as np
        out, R = import_adapter(a.adapter)
        np.savez(a.out, **out)
        print(f"[import] {len(out)} tensors rank={R} -> {a.out}")
        if a.selftest:
            selftest(a.out, R, a.adapter, "jax")
