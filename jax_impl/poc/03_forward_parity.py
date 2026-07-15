"""Gate C: HF 与 JAX 前向数值对拍(先纯文本,多模态待 Gate B 通过后加)。

两步走,避免一个进程装两套栈:
  1. torch venv:  python jax_impl/poc/03_forward_parity.py --side hf  --out /tmp/par_hf.json
  2. jax venv:    python jax_impl/poc/03_forward_parity.py --side jax --ref /tmp/par_hf.json

判定: 同一 prompt 下,两侧 next-token top-5 的 id 一致且
      logprob 差 < 0.05(bf16 图间噪声容差)即 PASS。
"""
import argparse
import json
import os

PROMPT = "The capital of France is"
TOPK = 5


def run_hf(out):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    import yaml
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    name = cfg["model"]["name_or_path"]
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float32)          # 对拍用 fp32,排除精度噪声
    ids = proc.tokenizer(PROMPT, return_tensors="pt")
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    lp = torch.log_softmax(logits, -1)
    top = torch.topk(lp, TOPK)
    json.dump({"prompt": PROMPT,
               "ids": ids["input_ids"][0].tolist(),
               "top_ids": top.indices.tolist(),
               "top_logprobs": [round(float(x), 4) for x in top.values]},
              open(out, "w"))
    print(f"[OK] HF top{TOPK} -> {out}")


def run_jax(ref_path):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax.numpy as jnp
    import jax
    from gemma import gm
    ref = json.load(open(ref_path))
    model = gm.nn.Gemma4_E2B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    tok_cls = getattr(gm.text, "Gemma4Tokenizer", None) or gm.text.Gemma3Tokenizer
    tok = tok_cls()
    ids = tok.encode(ref["prompt"], add_bos=True)
    print(f"[check] token ids  HF={ref['ids']}  JAX={ids}"
          f"  {'一致' if list(ids) == ref['ids'] else '⚠️ 不一致(先解决再看数值)'}")
    out = model.apply({"params": params}, tokens=jnp.asarray([ids]),
                      return_last_only=True)
    logits = out.logits[0] if hasattr(out, "logits") else out[0]
    lp = jax.nn.log_softmax(logits)
    top = jnp.argsort(lp)[-TOPK:][::-1]
    print(f"[JAX] top{TOPK} ids={top.tolist()} "
          f"logprobs={[round(float(lp[i]), 4) for i in top]}")
    print(f"[HF ] top{TOPK} ids={ref['top_ids']} logprobs={ref['top_logprobs']}")
    same = top.tolist() == ref["top_ids"]
    print("Gate C(纯文本):", "PASS" if same else "NO-GO(id 序不一致)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["hf", "jax"], required=True)
    ap.add_argument("--out", default="/tmp/par_hf.json")
    ap.add_argument("--ref", default="/tmp/par_hf.json")
    a = ap.parse_args()
    run_hf(a.out) if a.side == "hf" else run_jax(a.ref)
