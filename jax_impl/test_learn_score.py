"""可学习打分头(TOKEN_LEARN_SCORE)CPU 单测(务必 JAX_PLATFORMS=cpu,绝不碰 TPU)。

证明:
  (a) 关闭(TOKEN_LEARN_SCORE 未设、head=None): _compress_soft_tokens 的
      topk/dyn 输出与"改动前快照" _ref_learn_score.npz 逐位一致(max|Δ|==0)。
  (b) 开启 + b=0 暖启: 附加分≡0 → 选择与关闭时逐位一致(零冲击)。
  (c) 开启 + 非零头: training=True 时 jax.grad(标量 loss) 对 (A,b) 非零且有限
      (证明 STE 软门确实回传梯度); training=False 的前向 == 硬选前向
      (STE 前向不变)。
  (d) 静态形状: dyn 输出恒 [B, n*k, D]。

运行: docker exec tpu_train sh -c 'cd /workspace &&
      JAX_PLATFORMS=cpu python3 jax_impl/test_learn_score.py'
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # 双保险: 决不初始化 TPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from jax_impl.data import _compress_soft_tokens

_FAILS = []
_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "_ref_learn_score.npz")


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
    if not ok:
        _FAILS.append(name)


def _mk_head(D, r, seed, b_zero):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, r)).astype(np.float32) * (1.0 / D ** 0.5)
    b = (np.zeros((r, 1), np.float32) if b_zero
         else rng.standard_normal((r, 1)).astype(np.float32))
    return jnp.asarray(A), jnp.asarray(b)


# ------------------------- (a) 关闭 = 逐位快照一致 -----------------------
def test_off_byte_identical():
    print("(a) 关闭时 topk/dyn 与改动前快照逐位一致")
    os.environ.pop("TOKEN_LEARN_SCORE", None)
    z = np.load(_REF)
    t = jnp.asarray(z["t"])
    o_topk = np.asarray(_compress_soft_tokens(t, 16, "topk"))
    o_dyn = np.asarray(_compress_soft_tokens(t, 32, "dyn"))
    d1 = float(np.abs(o_topk - z["topk_k16"]).max())
    d2 = float(np.abs(o_dyn - z["dyn_k32"]).max())
    check("topk 关闭态 max|Δ|==0", d1 == 0.0, f"max|Δ|={d1:.3e}")
    check("dyn 关闭态 max|Δ|==0", d2 == 0.0, f"max|Δ|={d2:.3e}")


# ------------------------- (b) 开启 + b=0 = 暖启无冲击 -------------------
def test_warmstart_noop():
    print("(b) 开启 + b=0 暖启: 选择与关闭时逐位一致")
    os.environ["TOKEN_LEARN_SCORE"] = "1"
    os.environ.pop("TOKEN_LEARN_TRAIN", None)         # training=False 默认
    z = np.load(_REF)
    t = jnp.asarray(z["t"])
    D = t.shape[-1]
    for mode, k in (("topk", 16), ("dyn", 32)):
        head = _mk_head(D, 16, 7, b_zero=True)
        out = np.asarray(_compress_soft_tokens(t, k, mode, head=head))
        ref = z["topk_k16"] if mode == "topk" else z["dyn_k32"]
        d = float(np.abs(out - ref).max())
        check(f"{mode} b=0 暖启 max|Δ|==0(与关闭态同)", d == 0.0,
              f"max|Δ|={d:.3e}")
    os.environ.pop("TOKEN_LEARN_SCORE", None)


# ------------------- (c) 开启 + 非零头: 梯度回传 + 前向不变 --------------
def test_ste_gradient_and_invariance():
    print("(c) 开启 + 非零头: STE 梯度非零有限 + 前向不变")
    os.environ["TOKEN_LEARN_SCORE"] = "1"
    rng = np.random.default_rng(3)
    B, n, C, D = 2, 4, 64, 8
    t = jnp.asarray(rng.standard_normal((B, n, C, D)).astype(np.float32))
    A, b = _mk_head(D, 16, 21, b_zero=False)          # b 非零 → A 有梯度

    for mode, k in (("topk", 16), ("dyn", 32)):
        def loss(A_, b_):
            out = _compress_soft_tokens(t, k, mode, head=(A_, b_),
                                        training=True)
            return jnp.sum(out)
        gA, gb = jax.grad(loss, argnums=(0, 1))(A, b)
        gA, gb = np.asarray(gA), np.asarray(gb)
        finite = np.isfinite(gA).all() and np.isfinite(gb).all()
        nz = float(np.abs(gA).max()) > 0 and float(np.abs(gb).max()) > 0
        check(f"{mode} STE: d loss/d(A,b) 有限", bool(finite))
        check(f"{mode} STE: d loss/d(A,b) 非零", bool(nz),
              f"|gA|max={np.abs(gA).max():.3e} |gb|max={np.abs(gb).max():.3e}")

        # 前向不变: training=True 的前向 == training=False(纯硬选)
        o_tr = np.asarray(_compress_soft_tokens(t, k, mode, head=(A, b),
                                                training=True))
        o_hd = np.asarray(_compress_soft_tokens(t, k, mode, head=(A, b),
                                                training=False))
        d = float(np.abs(o_tr - o_hd).max())
        check(f"{mode} STE 前向不变 max|Δ|==0", d == 0.0, f"max|Δ|={d:.3e}")
    os.environ.pop("TOKEN_LEARN_SCORE", None)


# ------------------------- (d) 静态形状 --------------------------------
def test_shape():
    print("(d) dyn 静态形状 [B, n*k, D]")
    os.environ["TOKEN_LEARN_SCORE"] = "1"
    rng = np.random.default_rng(5)
    B, n, C, D, k = 3, 4, 64, 8, 32
    t = jnp.asarray(rng.standard_normal((B, n, C, D)).astype(np.float32))
    head = _mk_head(D, 16, 9, b_zero=False)
    out = _compress_soft_tokens(t, k, "dyn", head=head, training=True)
    check("dyn 形状 [B,n*k,D]", tuple(out.shape) == (B, n * k, D),
          str(tuple(out.shape)))
    os.environ.pop("TOKEN_LEARN_SCORE", None)


def main():
    print("=" * 60)
    print("learn-score CPU 单测(JAX_PLATFORMS=%s)"
          % os.environ.get("JAX_PLATFORMS"))
    print("=" * 60)
    test_off_byte_identical()
    test_warmstart_noop()
    test_ste_gradient_and_invariance()
    test_shape()
    print("=" * 60)
    if _FAILS:
        print(f"结果: FAIL —— {len(_FAILS)} 项未过: {_FAILS}")
        raise SystemExit(1)
    print("结果: ALL PASS")


if __name__ == "__main__":
    main()
