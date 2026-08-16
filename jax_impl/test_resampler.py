"""学习式重采样器(TOKEN_RESAMPLER)CPU 单测(务必 JAX_PLATFORMS=cpu,绝不碰 TPU)。

证明:
  (a) 纯函数前向: 随机 [1,16,64,8] → [1,512,8],全有限;1/2 层、有/无 FFN、
      多头切分均形状正确;batch 泛化 [3,...] → [3,512,8]。
  (b) 可导性: MSE(输出, 随机目标) 对全部参数叶的梯度有限且非零
      (重采样器=softmax 注意力,天然可导 —— 不同于硬 top_k 需 STE)。
  (c) 默认关零改动: env 未设时 make_vision_input 的 counts 与旧行为一致
      ((64,)*16),不安装补丁;_compress_soft_tokens 未被本次改动触碰
      (dyn 输出与历史快照 _ref_learn_score.npz 逐位一致)。
  (d) 参数确在可训练树: TOKEN_RESAMPLER=1 时 make_vision_input →
      install_token_select 的 patched setup 生效,真模型(Gemma4 E2B cfg64)
      jax.eval_shape(model.init) 的 struct["params"] 里出现全部 tok_resampler_*
      叶(顶层,形状与 _resampler_specs 一致)—— 与 train_sft 抽 train["tok"]
      / infer 注回用的是同一 struct 路径,故"补丁内参数可达+可训+可存取"。
  (e) 保存→加载往返: train["tok"] 以 "tok/<路径>" 存 npz(train_sft 同款
      flatten),经 npz_io.restore_train_tree 读回逐位一致。

运行: docker exec tpu_train sh -c 'cd /workspace &&
      JAX_PLATFORMS=cpu python3 jax_impl/test_resampler.py'
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # 双保险: 决不初始化 TPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_FAILS = []


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  {extra}" if extra else ""))
    if not ok:
        _FAILS.append(name)


def _mk_prm(D, total, layers, ffn, ffn_mult, seed=0):
    from jax_impl.data import _resampler_specs, resampler_init_leaf
    rng = np.random.RandomState(seed)
    return {name: resampler_init_leaf(name, shape, rng)
            for name, shape, _k in _resampler_specs(
                D, total, layers, ffn, ffn_mult)}


# ------------------- (a) 纯函数前向: 形状 / 有限性 -----------------------
def test_forward():
    print("(a) 前向: [1,16,64,8] → [1,512,8] 有限;层数/FFN/多头/批量变体")
    import jax.numpy as jnp
    from jax_impl.data import _resample_soft_tokens
    rng = np.random.default_rng(0)
    t = jnp.asarray(rng.standard_normal((1, 16, 64, 8)).astype(np.float32))
    for layers, ffn, heads in ((1, True, 2), (2, True, 2),
                               (1, False, 4), (2, False, 8)):
        prm = {k: jnp.asarray(v) for k, v in
               _mk_prm(8, 512, layers, ffn, 2, seed=layers).items()}
        out = np.asarray(_resample_soft_tokens(t, prm, heads))
        tag = f"L={layers} FFN={int(ffn)} H={heads}"
        check(f"形状 [1,512,8] ({tag})", out.shape == (1, 512, 8),
              str(out.shape))
        check(f"全有限 ({tag})", np.isfinite(out).all())
    t3 = jnp.asarray(rng.standard_normal((3, 16, 64, 8)).astype(np.float32))
    prm = {k: jnp.asarray(v) for k, v in _mk_prm(8, 512, 1, True, 2).items()}
    out3 = np.asarray(_resample_soft_tokens(t3, prm, 2))
    check("批量 [3,16,64,8] → [3,512,8]", out3.shape == (3, 512, 8))
    # 同输入同参确定性(无 dropout/随机源)
    out3b = np.asarray(_resample_soft_tokens(t3, prm, 2))
    check("前向确定性 max|Δ|==0",
          float(np.abs(out3 - out3b).max()) == 0.0)


# ------------------- (b) 可导性: 全叶梯度有限非零 ------------------------
def test_grads():
    print("(b) 可导性: MSE 对全部参数叶梯度有限且非零(无 STE 需求)")
    import jax
    import jax.numpy as jnp
    from jax_impl.data import _resample_soft_tokens
    rng = np.random.default_rng(1)
    t = jnp.asarray(rng.standard_normal((1, 16, 64, 8)).astype(np.float32))
    tgt = jnp.asarray(rng.standard_normal((1, 512, 8)).astype(np.float32))
    prm = {k: jnp.asarray(v) for k, v in _mk_prm(8, 512, 1, True, 2).items()}

    def loss(p):
        return jnp.mean((_resample_soft_tokens(t, p, 2) - tgt) ** 2)

    g = jax.grad(loss)(prm)
    bad_f = [k for k, v in g.items() if not np.isfinite(np.asarray(v)).all()]
    bad_z = [k for k, v in g.items() if float(np.abs(np.asarray(v)).max()) == 0]
    check("全部叶梯度有限", not bad_f, str(bad_f))
    check("全部叶梯度非零", not bad_z, str(bad_z))


# ------------------- (c) 默认关: 现有路径零改动 --------------------------
def test_default_off():
    print("(c) 默认关: counts 旧口径 + _compress_soft_tokens 快照逐位一致")
    for k in ("TOKEN_RESAMPLER", "SELECT_TOKENS_K", "TOKEN_COMPRESS_MODE",
              "TOKEN_LEARN_SCORE"):
        os.environ.pop(k, None)
    import jax.numpy as jnp
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    from jax_impl.data import make_vision_input, _compress_soft_tokens
    rng = np.random.default_rng(2)
    frames = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
              for _ in range(16)]
    _pa, _px, counts = make_vision_input([frames])
    check("env 关 → counts=(64,)*16", counts == (64,) * 16, str(counts))
    check("env 关 → 未安装补丁",
          not getattr(g4_tr, "_TOKEN_SELECT_PATCHED", False))
    ref = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_ref_learn_score.npz")
    z = np.load(ref)
    o_dyn = np.asarray(_compress_soft_tokens(jnp.asarray(z["t"]), 32, "dyn"))
    d = float(np.abs(o_dyn - z["dyn_k32"]).max())
    check("dyn 快照逐位一致(压缩核心未被触碰)", d == 0.0, f"max|Δ|={d:.3e}")
    return frames


# --------------- (d) 真模型可训练树 + (e) 保存/加载往返 ------------------
def test_model_tree_and_roundtrip(frames):
    print("(d) TOKEN_RESAMPLER=1: 真模型 struct 出现 tok_resampler_* 叶")
    os.environ["TOKEN_RESAMPLER"] = "1"          # RESAMPLER_* 全走默认
    import dataclasses
    import jax
    import jax.numpy as jnp
    from gemma import gm
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
    from jax_impl.data import make_vision_input, _resampler_specs

    p0, x0, counts = make_vision_input([frames])  # 触发补丁安装
    check("counts=(512,) 单块(dyn 同款口径)", counts == (512,), str(counts))
    check("补丁已安装", getattr(g4_tr, "_TOKEN_SELECT_PATCHED", False))
    check("重采样器 setup 已补",
          getattr(g4_tr, "_TOK_RESAMPLER_PATCHED", False))

    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    model = gm.nn.Gemma4_E2B(text_only=False, config=cfg64)
    pvi = PreprocessedVisionInput(
        patches=jnp.asarray(p0), positions_xy=jnp.asarray(x0),
        soft_token_counts=counts)
    toks = np.zeros((1, 900), np.int32)
    toks[0, 300:812] = -2                        # 512 个视觉哨兵位
    struct = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0), tokens=jnp.asarray(toks), images=pvi))

    flat = {"/".join(getattr(kk, "key", str(kk)) for kk in path): leaf
            for path, leaf in
            jax.tree_util.tree_flatten_with_path(struct["params"])[0]}
    got = {k: tuple(v.shape) for k, v in flat.items() if "tok_resampler" in k}
    D = cfg64.vision_encoder.d_model
    want = {name: shape for name, shape, _k in
            _resampler_specs(D, 512, 1, True, 2)}   # 默认 L1/FFN1/mult2
    check(f"struct 含全部 {len(want)} 个 tok_resampler_* 叶(D={D})",
          got == {k: tuple(v) for k, v in want.items()},
          f"got={len(got)}叶")
    n_par = sum(int(np.prod(s)) for s in want.values())
    print(f"  [info] 重采样器参数量(真 D={D}): {n_par/1e6:.2f}M "
          f"(fp32 ≈{n_par*4/2**20:.0f}MB)")

    # train_sft 同款抽树: struct 顶层路径 → train["tok"] = {路径: 初值}
    from jax_impl.data import resampler_init_leaf
    rng = np.random.RandomState(0)
    tok0 = {ps: resampler_init_leaf(ps, leaf.shape, rng)
            for ps, leaf in flat.items() if "tok_resampler" in ps}
    check("train['tok'] 抽树 = struct 全叶", set(tok0) == set(want))
    # loss_fn 同款注入: 沿路径写回 merged params(此处用假顶层 dict 验证路径)
    fake = {k: None for k in flat if "tok_resampler" not in k}
    for ps, leaf in tok0.items():
        segs = ps.split("/")
        node = fake
        for s in segs[:-1]:
            node[s] = dict(node[s])
            node = node[s]
        node[segs[-1]] = leaf
    check("注入后 merged 树含全叶",
          all(isinstance(fake.get(k.split('/')[0]), np.ndarray)
              for k in tok0))

    print("(e) 保存→加载往返(npz 'tok/<路径>',train_sft/infer 同款键)")
    import tempfile
    from jax_impl.npz_io import restore_train_tree
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "warm.npz")
        np.savez(f, **{"tok/" + k: v for k, v in tok0.items()})
        z = np.load(f)
        check("npz 键样例 tok/tok_resampler_q",
              "tok/tok_resampler_q" in z.files)
        # restore_train_tree 走与 train_sft --init-npz 完全相同的路径匹配
        blank = {"tok": {k: jnp.zeros(v.shape, jnp.float32)
                         for k, v in tok0.items()}}
        got_tree, st = restore_train_tree(
            blank, z, jnp, is_zero_skippable=lambda _p: False)
        check(f"restore 命中全部 {len(tok0)} 叶", st["hit"] == len(tok0),
              str(st))
        diffs = [float(np.abs(np.asarray(got_tree["tok"][k]) - tok0[k]).max())
                 for k in tok0]
        check("往返逐位一致 max|Δ|==0", max(diffs) == 0.0,
              f"max|Δ|={max(diffs):.3e}")
    os.environ.pop("TOKEN_RESAMPLER", None)


def main():
    print("=" * 60)
    print("resampler CPU 单测(JAX_PLATFORMS=%s)"
          % os.environ.get("JAX_PLATFORMS"))
    print("=" * 60)
    test_forward()
    test_grads()
    frames = test_default_off()
    test_model_tree_and_roundtrip(frames)
    print("=" * 60)
    if _FAILS:
        print(f"结果: FAIL —— {len(_FAILS)} 项未过: {_FAILS}")
        raise SystemExit(1)
    print("结果: ALL PASS")


if __name__ == "__main__":
    main()
