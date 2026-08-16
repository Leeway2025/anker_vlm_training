"""tome(时空 ToMe 双部软匹配)CPU 单测(务必 JAX_PLATFORMS=cpu,绝不碰 TPU)。
覆盖:
  (a) 设备 tome 分支形状:输出恰 TOME_TOTAL 个 token、跨 seed/batch 静态恒定;
  (b) 段均值语义:合并 token == 其组(dst B + 并入 A)的均值(与 numpy 参照逐位一致);
  (c) 跨帧合并确实发生:合成"多帧同背景"输入,大量 A 归并到【另一帧】的 B;
  (d) n_keep>0(total>Nb)保留分支:形状 = merged(Nb) + 保留 A(total-Nb),数值一致;
  (e) mode-OFF:topk 路径与 numpy 经典 Top-K 参照逐位一致(未被 tome 改动)。
运行: docker exec tpu_train bash -lc 'cd /workspace &&
       JAX_PLATFORMS=cpu python3 jax_impl/test_tome.py'
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # 双保险:决不初始化 TPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from jax_impl.data import _compress_soft_tokens

_FAILS = []


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
    if not ok:
        _FAILS.append(name)


def _tome(t, total):
    import jax.numpy as jnp
    return np.asarray(_compress_soft_tokens(
        jnp.asarray(t), 0, "tome", total=total))


def _np_tome(t, total):
    """numpy 参照:严格复刻设备 tome(双部偶/奇切分→余弦→argmax→段均值)。"""
    B, n, C, D = t.shape
    N = n * C
    tf = t.reshape(B, N, D)
    pair = tf.reshape(B, N // 2, 2, D)
    A, Bs = pair[:, :, 0, :], pair[:, :, 1, :]
    Na = Nb = N // 2
    n_keep = total - Nb
    An = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-6)
    Bn = Bs / (np.linalg.norm(Bs, axis=-1, keepdims=True) + 1e-6)
    sim = np.einsum("bad,bnd->ban", An, Bn)
    dst = sim.argmax(-1)
    out = []
    for b in range(B):
        oh = np.zeros((Na, Nb), np.float32)
        oh[np.arange(Na), dst[b]] = 1.0
        keep_idx = None
        if n_keep > 0:
            edge = sim[b].max(-1)
            keep_idx = np.sort(np.argsort(edge)[:n_keep])   # edge 最低=最独特
            oh[keep_idx, :] = 0.0
        contrib = np.einsum("ad,an->nd", A[b], oh)
        count = oh.sum(0)
        merged = (Bs[b] + contrib) / (1.0 + count[:, None])
        if n_keep > 0:
            out.append(np.concatenate([merged, A[b][keep_idx]], 0))
        else:
            out.append(merged)
    return np.stack(out)


# --------------------------- (a) 形状静态性 -----------------------------
def test_shape():
    print("(a) tome 输出形状静态恒定")
    rng = np.random.default_rng(0)
    # 目标消融口径:16 帧 × 64 = 1024 源 → 合并 512
    for seed in range(3):
        rg = np.random.default_rng(seed)
        for B in (1, 2, 3):
            t = rg.standard_normal((B, 16, 64, 8)).astype(np.float32)
            o = _tome(t, 512)
            check(f"[seed{seed} B{B}] 输出 [B,512,D]",
                  o.shape == (B, 512, 8), str(o.shape))
    # 跨 batch/seed 第二维恒为 total,与内容无关(静态形状)
    shapes = {(_tome(np.random.default_rng(s).standard_normal(
        (2, 16, 64, 8)).astype(np.float32), 512).shape[1]) for s in range(4)}
    check("第二维跨 seed 恒 == 512", shapes == {512}, str(shapes))


# --------------------------- (b) 段均值语义 -----------------------------
def test_mean_semantics():
    print("(b) 合并 token == 组均值(与 numpy 参照逐位一致)")
    rng = np.random.default_rng(1)
    B, n, C, D = 2, 4, 8, 16          # N=32,half=16
    t = rng.standard_normal((B, n, C, D)).astype(np.float32)  # 连续值→无并列
    o = _tome(t, 16)                 # total==Nb → 全并、n_keep=0
    exp = _np_tome(t, 16)
    check("total==Nb 输出形状", o.shape == (B, 16, D), str(o.shape))
    check("与 numpy 段均值参照逐位一致", np.allclose(o, exp, atol=1e-5),
          f"max|Δ|={np.abs(o - exp).max():.2e}")

    # 显式手算:构造 A 全部并入同一个 B(让所有 A 指向 B 的 0 号列)
    t2 = np.zeros((1, 2, 4, 4), np.float32)   # N=8,half=4;A=[g0,g2,g4,g6]
    # B 列 0(全局 idx1)设成大范数唯一方向,其余 B 列范数极小→所有 A 余弦最偏向列0
    tf = t2.reshape(1, 8, 4)
    e0 = np.array([1, 0, 0, 0], np.float32)
    for a in (0, 2, 4, 6):
        tf[0, a] = e0 * (a + 1)               # A 方向都 ∝ e0
    tf[0, 1] = e0 * 5.0                        # B 列0:同向大范数(argmax 命中)
    for b in (3, 5, 7):
        tf[0, b] = np.array([0, 1, 0, 0], np.float32) * 1e-3  # 其余 B 近零、正交
    t2 = tf.reshape(1, 2, 4, 4)
    o2 = _tome(t2, 4)
    grp = np.stack([tf[0, 1], tf[0, 0], tf[0, 2], tf[0, 4], tf[0, 6]])
    check("全并入同一 B → 该 token==(B0+4个A)/5",
          np.allclose(o2[0, 0], grp.mean(0), atol=1e-5),
          f"got={o2[0,0].tolist()} exp={grp.mean(0).tolist()}")


# --------------------------- (c) 跨帧合并 -------------------------------
def test_cross_frame():
    print("(c) 跨帧合并确实发生(合成多帧同背景)")
    n, C, D = 4, 8, 16
    N = n * C
    rng = np.random.default_rng(2)
    base = rng.standard_normal((C, D)).astype(np.float32)   # 逐位置背景原型
    # 每帧 = 同一背景(跨帧冗余),叠加极小逐帧噪声(打破完全并列但保持最相似跨帧)
    t = np.stack([base + 1e-4 * rng.standard_normal((C, D)).astype(np.float32)
                  for _ in range(n)])[None]                 # [1,n,C,D]
    import jax.numpy as jnp
    # 复算 dst 判定跨帧:A 全局偶位,B 全局奇位
    tf = t.reshape(1, N, D)
    pair = tf.reshape(1, N // 2, 2, D)
    A, Bs = pair[0, :, 0, :], pair[0, :, 1, :]
    An = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-6)
    Bn = Bs / (np.linalg.norm(Bs, axis=-1, keepdims=True) + 1e-6)
    dst = (An @ Bn.T).argmax(-1)
    a_glob = np.arange(0, N, 2)                    # A 的全局 idx(偶)
    b_glob = 2 * dst + 1                           # 命中 B 的全局 idx(奇)
    a_frame, b_frame = a_glob // C, b_glob // C
    cross = int((a_frame != b_frame).sum())
    check("存在跨帧合并(dst 帧 != src 帧)", cross > 0,
          f"跨帧 {cross}/{len(a_glob)}")
    check("跨帧合并占多数(冗余背景被跨时间折叠)", cross >= len(a_glob) // 2,
          f"跨帧 {cross}/{len(a_glob)}")
    # 输出仍是恰 N/2 且数值 == 参照
    o = _tome(t, N // 2)
    check("背景合并输出 [1,N/2,D]", o.shape == (1, N // 2, D), str(o.shape))
    check("背景合并与参照一致", np.allclose(o, _np_tome(t, N // 2), atol=1e-4))


# --------------------------- (d) n_keep>0 保留分支 ----------------------
def test_keep_branch():
    print("(d) total>Nb 保留分支(merged Nb + 保留 A)")
    rng = np.random.default_rng(3)
    B, n, C, D = 2, 4, 8, 16          # N=32,Nb=16
    t = rng.standard_normal((B, n, C, D)).astype(np.float32)
    total = 20                        # n_keep=4
    o = _tome(t, total)
    check("输出 [B,total,D]", o.shape == (B, total, D), str(o.shape))
    check("与 numpy 保留分支参照逐位一致",
          np.allclose(o, _np_tome(t, total), atol=1e-5),
          f"max|Δ|={np.abs(o - _np_tome(t, total)).max():.2e}")


# --------------------------- (e) mode-OFF:topk 不变 --------------------
def _np_score(t):
    motion = np.linalg.norm(t - t.mean(axis=1, keepdims=True), axis=-1)
    norm = np.linalg.norm(t, axis=-1)

    def z(x):
        return (x - x.mean(-1, keepdims=True)) / (x.std(-1)[..., None] + 1e-6)
    return z(motion) + 0.5 * z(norm)


def test_topk_unchanged():
    print("(e) mode-OFF:topk 路径未被 tome 改动")
    rng = np.random.default_rng(4)
    B, n, C, D = 2, 6, 64, 16
    k = 32
    t = rng.standard_normal((B, n, C, D)).astype(np.float32)
    import jax.numpy as jnp
    o = np.asarray(_compress_soft_tokens(jnp.asarray(t), k, "topk"))
    check("topk 输出 [B,n*k,D]", o.shape == (B, n * k, D), str(o.shape))
    # numpy 参照:逐帧取分数最高 k(位置),按光栅升序 gather
    score = _np_score(t)
    exp = np.empty((B, n * k, D), np.float32)
    for b in range(B):
        for i in range(n):
            idx = np.sort(np.argsort(-score[b, i], kind="stable")[:k])
            exp[b, i * k:(i + 1) * k] = t[b, i, idx]
    check("topk 与经典 Top-K 参照逐位一致", np.allclose(o, exp, atol=1e-5),
          f"max|Δ|={np.abs(o - exp).max():.2e}")


def main():
    print("=" * 60)
    print("tome CPU 单测(JAX_PLATFORMS=%s)" % os.environ.get("JAX_PLATFORMS"))
    print("=" * 60)
    test_shape()
    test_mean_semantics()
    test_cross_frame()
    test_keep_branch()
    test_topk_unchanged()
    print("=" * 60)
    if _FAILS:
        print(f"结果: FAIL —— {len(_FAILS)} 项未过: {_FAILS}")
        raise SystemExit(1)
    print("结果: ALL PASS")


if __name__ == "__main__":
    main()
