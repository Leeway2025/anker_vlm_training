"""SWA 权重平均(JAX 产物 npz 版)。

  python jax_impl/swa.py --ckpts out1/train_params.npz out2/... --out DIR
平均所有输入 npz 的同名叶(形状一致才平均,否则报错不静默)。
产物仍是 npz,可 --init-npz 续训或 export_hf.py 导出。
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zs = [np.load(p) for p in a.ckpts]
    common = set(zs[0].files)
    union = set(zs[0].files)
    for z in zs[1:]:
        common &= set(z.files); union |= set(z.files)
    if union - common:
        frac = len(union - common) / max(len(union), 1)
        print(f"[swa] ⚠️ {len(union-common)}/{len(union)} 键非共有"
              f"({frac:.0%}),取首个 ckpt 原值 —— 树不同构。")
        if frac > 0.3:
            print("[swa] ⚠️⚠️ 异构检查点平均(如 SFT 全量树 × KTO 纯 LoRA "
                  "树)不是标准 SWA 用法,实测会掉点(FINDINGS v8 R7);"
                  "标准用法 = 同一训练轨迹的邻近检查点。三思。")
    avg = {}
    for k in union:
        arrs = [z[k] for z in zs if k in z.files]
        assert all(x.shape == arrs[0].shape for x in arrs), f"形状不齐: {k}"
        avg[k] = (np.mean(arrs, axis=0).astype(arrs[0].dtype)
                  if k in common else arrs[0])
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "train_params.npz"), **avg)
    print(f"[swa] {len(a.ckpts)} ckpts × {len(avg)} tensors -> {a.out}")


if __name__ == "__main__":
    main()
