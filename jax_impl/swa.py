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
        print(f"[swa] ⚠️ {len(union-common)} 个非共有键取首个 ckpt 的值"
              f"(树不同构: 如有无 proj/aux)")
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
