"""projector 双向转换: torch projector.pt ↔ JAX npz(跨框架 stage a 衔接)。

  torch → JAX(把 torch stage a 的成果喂给 JAX 5b 的 --init-npz):
    <torch_venv>/bin/python jax_impl/convert_projector.py \
        --torch-pt outputs/phase5_sft_a/final/projector.pt --out proj_a.npz
  JAX → torch(把 JAX 训过的 projector 交回 torch/RKLLM 生态):
    <torch_venv>/bin/python jax_impl/convert_projector.py \
        --npz outputs/jax_5b/train_params.npz --out projector.pt

方向已用 base 权重逐位证明(2026-07-16 实测):
  JAX embedder/mm_input_projection/w (768,1536)
  = HF model.embed_vision.embedding_projection.weight (1536,768) 的转置,
  maxdiff 0.0014(bf16 舍入级)。
运行环境: torch venv(读写 .pt 需要 torch;npz 用 numpy)。
自检: 每次转换都做回环(转过去再转回来,逐位一致才落盘)。
"""
import argparse
import os

import numpy as np

JAX_KEY = "proj/mm_input_projection/w"          # train_sft --init-npz 的恢复键
PT_KEY_SUB = "embed_vision.embedding_projection.weight"


def torch_to_jax(pt_path, out_path):
    import torch
    sd = torch.load(pt_path, map_location="cpu")
    keys = [k for k in sd if PT_KEY_SUB in k]
    assert keys, f"{pt_path} 中找不到 *{PT_KEY_SUB}(现有键: {list(sd)[:5]})"
    W = sd[keys[0]].float().numpy()             # [1536, 768]
    w = np.ascontiguousarray(W.T)               # → (768, 1536)
    assert np.abs(w.T - W).max() == 0           # 回环自检
    np.savez(out_path, **{JAX_KEY: w})
    print(f"[OK] {keys[0]} {W.shape} → {JAX_KEY} {w.shape} -> {out_path}")
    print("用法: train_sft.py 加 --init-npz", out_path, "(--train-projector 保持开)")


def jax_to_torch(npz_path, out_path, key_style="peft"):
    import torch
    z = np.load(npz_path)
    # 只认 proj/ 子树 —— 泛匹配会误抓 embedder 上的 LoRA 键(真踩过:
    # lora/embedder/mm_input_projection/_LoRAEinsum_0/lora/a (768,16))
    cands = [k for k in z.files
             if k == JAX_KEY or (k.startswith("proj/")
                                 and "mm_input_projection" in k)]
    assert cands, (f"{npz_path} 中无 projector 权重键(proj/...)。"
                   f"训练时开了 --train-projector 吗?")
    w = z[cands[0]].astype(np.float32)
    assert w.shape == (768, 1536), f"projector 形状异常: {w.shape}"
    W = np.ascontiguousarray(w.T)               # → [1536, 768]
    assert np.abs(W.T - w).max() == 0           # 回环自检
    prefix = ("base_model.model.model." if key_style == "peft" else "model.")
    key = prefix + PT_KEY_SUB
    torch.save({key: torch.from_numpy(W).to(torch.bfloat16)}, out_path)
    print(f"[OK] {cands[0]} {w.shape} → {key} {W.shape} -> {out_path}")
    print("用法: torch train.py --init-from 目录中放置此 projector.pt,"
          "或 restore_from 直接加载(strict=False)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--torch-pt", help="输入 torch projector.pt → 输出 npz")
    g.add_argument("--npz", help="输入 JAX train_params.npz → 输出 projector.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-style", choices=["peft", "plain"], default="peft",
                    help="jax→torch 时的键前缀(peft=与训练产物一致)")
    a = ap.parse_args()
    if a.torch_pt:
        torch_to_jax(a.torch_pt, a.out)
    else:
        jax_to_torch(a.npz, a.out, a.key_style)
