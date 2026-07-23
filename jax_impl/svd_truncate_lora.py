"""大 rank LoRA 产物 → SVD 截断为小 rank 标准 LoRA(零重训压缩)。

  python3 jax_impl/svd_truncate_lora.py --in outputs/jax_5b_s4/train_params_best.npz \
      --rank 64 --out outputs/jax_5b_s4/train_params_r64.npz
  python3 jax_impl/svd_truncate_lora.py --in ... --report-only   # 只看奇异谱

原理: prod 前向 out += scale(r)·x@a@b(rsLoRA scale=2√r)。对每个
ΔW = scale·a@b 做 SVD,保留前 k 个奇异方向,重分解为
a' = U_k·√Σ_k, b' = √Σ_k·V_kᵀ —— **scale 已折进因子**,产物为
uniform 单一 rank,加载器(detect_rank_scheme)自动判 uniform、
前向 scale=1,数学上等价于最优 rank-k 近似。

产出物:
  --out npz     lora/ 全部截断为 rank k;proj/ 等其余子树原样透传
  能量报表      每矩阵 top-k 奇异值能量占比(均值/最差)——
                占比高 ⇒ 大 rank 冗余(100k 数据未用满)的直接证据

验收纪律(用前必做):
  ① --rank 0(满秩重分解)产物评测必须与原产物**逐分对齐**(回环门禁);
  ② 截断版评测对照原版,掉 ≤0.5 可直接交付,掉多走蒸馏修复。
纯 numpy/stdlib,宿主机可跑(无需 TPU/jax)。
"""
import argparse
import math

import numpy as np

E2B_GLOBAL_LAYERS = frozenset({4, 9, 14, 19, 24, 29, 34})


def prod_scale_for_key(key, a_shape):
    """与 prod_lora.install_prod_lora 同源的 scale 判定(α=2r,rsLoRA)。"""
    r = int(a_shape[-1])
    return 2.0 * r / math.sqrt(r)          # = 2*sqrt(r)


def truncate_pair(a, b, scale, k):
    """(a: […, r], b: [r, …]) → rank-k 最优近似的 (a', b', 能量占比)。
    scale 折进因子;k=0 表示满秩重分解(回环验证用)。"""
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    in_dims, r = a64.shape[:-1], a64.shape[-1]
    out_dims = b64.shape[1:]
    A2 = a64.reshape(-1, r)                       # (In, r)
    B2 = b64.reshape(r, -1)                       # (r, Out)
    dw = scale * (A2 @ B2)                        # (In, Out)
    u, s, vt = np.linalg.svd(dw, full_matrices=False)
    k_eff = min(k or len(s), len(s))
    energy = float((s[:k_eff] ** 2).sum() / max((s ** 2).sum(), 1e-30))
    root = np.sqrt(s[:k_eff])
    a_new = (u[:, :k_eff] * root).reshape(*in_dims, k_eff)
    b_new = (root[:, None] * vt[:k_eff]).reshape(k_eff, *out_dims)
    return a_new, b_new, energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank", type=int, default=64,
                    help="目标 rank;0 = 满秩重分解(回环门禁用)")
    ap.add_argument("--report-only", action="store_true",
                    help="只打印奇异谱能量报表,不写产物")
    a = ap.parse_args()
    if not a.report_only and not a.out:
        ap.error("--out 必填(或改用 --report-only)")

    z = np.load(a.inp)
    a_keys = sorted(k for k in z.files
                    if k.startswith("lora/") and k.endswith("/a"))
    if not a_keys:
        raise SystemExit("npz 中无 lora/…/a 键,不是训练产物")

    out, report = {}, []
    for ka in a_keys:
        kb = ka[:-2] + "/b"
        if kb not in z.files:
            raise SystemExit(f"缺配对键 {kb}")
        av, bv = z[ka], z[kb]
        if av.shape[-1] != bv.shape[0]:
            raise SystemExit(f"{ka} rank 轴不匹配: a{av.shape} b{bv.shape}")
        if np.abs(av).max() == 0 and np.abs(bv).max() == 0:
            # 未训练的死叶(如 embedder 占位): 原样透传截断形状的零
            k_eff = min(a.rank or av.shape[-1], av.shape[-1])
            out[ka] = np.zeros((*av.shape[:-1], k_eff), av.dtype)
            out[kb] = np.zeros((k_eff, *bv.shape[1:]), bv.dtype)
            continue
        scale = prod_scale_for_key(ka, av.shape)
        a_new, b_new, energy = truncate_pair(av, bv, scale, a.rank)
        out[ka] = a_new.astype(np.float32)
        out[kb] = b_new.astype(np.float32)
        report.append((ka[:-2], av.shape[-1], out[ka].shape[-1], energy))

    for k in z.files:                     # proj/ 与其他子树原样透传
        if not k.startswith("lora/"):
            out[k] = z[k]

    print(f"{'矩阵':<58}{'r':>5}{'→k':>5}{'能量保留':>9}")
    for name, r0, k_, e in sorted(report, key=lambda x: x[3]):
        print(f"{name:<58}{r0:>5}{k_:>5}{e:>9.2%}")
    es = [e for *_, e in report]
    print(f"\n共 {len(report)} 对;能量保留 均值 {np.mean(es):.2%} / "
          f"最差 {min(es):.2%} / 中位 {np.median(es):.2%}")
    print("判读: 均值 >95% ⇒ 大 rank 在当前数据量下冗余明显,截断损失可控;"
          "最差 <80% 的矩阵是掉分嫌疑,考虑蒸馏修复或该矩阵保大 rank")

    if a.report_only:
        return
    np.savez(a.out, **out)
    sz = sum(v.nbytes for k_, v in out.items() if k_.startswith("lora/"))
    print(f"[OK] → {a.out}(lora 子树 {sz/2**20:.0f} MB, float32);"
          f"评测: infer 对产物照常跑,加载器自动判 uniform r={a.rank or '满秩'}")


if __name__ == "__main__":
    main()
