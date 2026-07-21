"""训练产物 npz 体检: LoRA A/B 与 projector 的范数统计(判训练是否真更新)。

  python3 jax_impl/npz_stats.py --npz outputs/jax_5b/train_params_best.npz

判读: lora .../b 的 |max| 全部 ≈0 → LoRA 从未被更新(优化器/掩码问题),
推理等于 base+prompt;b 健康量级(1e-3~1e-1)→ 训练确实写入了 LoRA。
纯 numpy。
"""
import argparse
import collections

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    a = ap.parse_args()
    z = np.load(a.npz)
    grp = collections.defaultdict(lambda: [0, 0.0, 0.0])   # n, max, sumsq
    for k in z.files:
        v = z[k]
        if k.startswith("lora/"):
            tag = ("lora/vision" if "vision_encoder" in k else "lora/llm")
            tag += "/b" if k.endswith("/b") else "/a"
        elif k.startswith("proj/"):
            tag = "proj"
        elif k.startswith("aux/"):
            tag = "aux"
        else:
            tag = "other"
        g = grp[tag]
        g[0] += 1
        g[1] = max(g[1], float(np.abs(v).max()))
        g[2] += float((v.astype(np.float64) ** 2).sum())
    print(f"{'子树':<16}{'叶数':>6}{'|max|':>12}{'RMS':>12}")
    for tag in sorted(grp):
        n, mx, ss = grp[tag]
        size = sum(z[k].size for k in z.files
                   if (tag.startswith("lora/vision") and "vision_encoder" in k
                       and k.endswith("/" + tag[-1]))
                   or (tag == "lora/llm/a" and k.startswith("lora/")
                       and "vision_encoder" not in k and k.endswith("/a"))
                   or (tag == "lora/llm/b" and k.startswith("lora/")
                       and "vision_encoder" not in k and k.endswith("/b"))
                   or (tag == "proj" and k.startswith("proj/"))
                   or (tag == "aux" and k.startswith("aux/"))
                   or (tag == "other" and not any(
                       k.startswith(p) for p in ("lora/", "proj/", "aux/"))))
        rms = (ss / max(size, 1)) ** 0.5
        verdict = ""
        if tag.endswith("/b"):
            verdict = ("  ❌ B≈0: LoRA 未被训练!" if mx < 1e-6
                       else "  ✅ 已更新")
        print(f"{tag:<16}{n:>6}{mx:>12.3e}{rms:>12.3e}{verdict}")


if __name__ == "__main__":
    main()
