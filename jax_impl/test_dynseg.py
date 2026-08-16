"""dynseg CPU 单测(务必 JAX_PLATFORMS=cpu 运行,绝不碰 TPU)。
覆盖三层:
  (a) 宿主预算 _dynseg_budget:和==TOTAL、∈[FLOOR,64]、高梯度帧多得、
      全局 pan(机位运动)不虚高;
  (b) 设备 dynseg 分支:形状 [B,TOTAL,D]、保留数==TOTAL、帧内光栅序、
      预算大的帧贡献多、与 numpy 参照选择逐位一致;
  (c) 逐样本模板:不同预算下 T 恒定、哨兵总数==TOTAL、delim 数==n-1、
      逐帧哨兵游程==counts。
运行: docker exec tpu_train bash -lc 'cd /workspace &&
       JAX_PLATFORMS=cpu python3 jax_impl/test_dynseg.py'
"""
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # 双保险:决不初始化 TPU
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from jax_impl.data import (_dynseg_budget, _dynseg_template,
                           _compress_soft_tokens, SENTINEL)

_FAILS = []


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  {extra}" if extra else ""))
    if not ok:
        _FAILS.append(name)


# --------------------------- (a) 宿主预算 ---------------------------------
def _flat(v, h=48, w=48):
    return np.full((h, w, 3), v, np.uint8)


def _texture(seed, h=48, w=48):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_budget():
    print("(a) _dynseg_budget")
    TOTAL, FLOOR, n = 512, 8, 16
    # 和 & 界:随机帧序
    frames = [_texture(i) for i in range(n)]
    c = _dynseg_budget(frames, TOTAL, FLOOR)
    check("sum==TOTAL", int(c.sum()) == TOTAL, f"sum={int(c.sum())}")
    check("每项∈[FLOOR,64]", c.min() >= FLOOR and c.max() <= 64,
          f"min={c.min()} max={c.max()}")
    check("len==n", len(c) == n)

    # 高梯度帧 > 平坦帧:帧0 高频纹理,其余平坦
    TOTAL2, n2 = 256, 8
    detail = [_texture(99)] + [_flat(128) for _ in range(n2 - 1)]
    cd = _dynseg_budget(detail, TOTAL2, FLOOR)
    uniform = TOTAL2 // n2                                # 均匀份额=32
    check("高梯度帧预算 > 平坦帧", cd[0] > cd[1], f"c0={cd[0]} c1={cd[1]}")
    check("高梯度帧 > 均匀份额(把预算从平坦帧夺走)", cd[0] > uniform,
          f"c0={cd[0]} uniform={uniform}")
    check("平坦帧 <= 均匀份额", cd[1:].max() <= uniform,
          f"flat_max={cd[1:].max()} uniform={uniform}")
    check("detail sum==TOTAL", int(cd.sum()) == TOTAL2)

    # 机位鲁棒性:同一静态纹理仅做全局平移(pan)——不应让 pan 帧预算暴涨
    base = _texture(7)
    pan = [np.roll(base, shift=3 * i, axis=1) for i in range(n2)]  # 全局横移
    cp = _dynseg_budget(pan, TOTAL2, FLOOR)
    spread = int(cp.max() - cp.min())
    check("pan 序列预算近均匀(机位鲁棒)", spread <= 6,
          f"max-min={spread} counts={cp.tolist()}")
    # 定量对照:pan 帧最大预算 << 真高梯度帧预算
    check("pan 最大预算 < 高梯度帧预算", cp.max() < cd[0],
          f"pan_max={cp.max()} detail_max={cd[0]}")
    check("pan 预算方差 < detail 预算方差",
          float(np.std(cp)) < float(np.std(cd)),
          f"std_pan={np.std(cp):.2f} std_detail={np.std(cd):.2f}")

    # 边界:total==n*floor → 全 floor;total==n*64 → 全 64
    c_lo = _dynseg_budget(frames, n * FLOOR, FLOOR)
    check("total==n*floor → 全 floor", (c_lo == FLOOR).all())
    c_hi = _dynseg_budget(frames, n * 64, FLOOR)
    check("total==n*64 → 全 64", (c_hi == 64).all())


# --------------------------- (b) 设备 dynseg 分支 ------------------------
def _np_score(t):
    """numpy 复刻设备打分 z(motion)+0.5 z(norm),用于参照选择。"""
    motion = np.linalg.norm(t - t.mean(axis=1, keepdims=True), axis=-1)
    norm = np.linalg.norm(t, axis=-1)

    def z(x):
        return (x - x.mean(-1, keepdims=True)) / (x.std(-1) [..., None] + 1e-6)
    return z(motion) + 0.5 * z(norm)


def _np_expected(t, seg):
    """参照:逐帧取分数最高 seg[b,i] 个(位置),按 (帧,光栅) 序 gather。"""
    B, n, C, D = t.shape
    score = _np_score(t)
    out = []
    for b in range(B):
        rows = []
        for i in range(n):
            kk = int(seg[b, i])
            # 名次<kk 的位置(与设备 argsort(argsort) 同义),按原始光栅升序
            order = np.argsort(-score[b, i], kind="stable")
            rank = np.empty(C, int)
            rank[order] = np.arange(C)
            keep = np.nonzero(rank < kk)[0]           # 已是升序光栅
            rows.append(t[b, i, keep])
        out.append(np.concatenate(rows, axis=0))
    return np.stack(out)


def test_device():
    print("(b) _compress_soft_tokens(dynseg)")
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    B, n, C, D = 2, 8, 64, 16
    TOTAL = 256
    t = rng.standard_normal((B, n, C, D)).astype(np.float32)
    # 两个样本不同预算分布(和都==TOTAL)
    seg0 = _dynseg_budget([_texture(i) for i in range(n)], TOTAL, 8)
    seg1 = np.array([64, 40, 32, 24, 24, 24, 24, 24], np.int32)
    assert seg1.sum() == TOTAL
    seg = np.stack([seg0, seg1]).astype(np.int32)

    out = np.asarray(_compress_soft_tokens(
        jnp.asarray(t), 0, "dynseg", seg=jnp.asarray(seg), total=TOTAL))
    check("输出形状 [B,TOTAL,D]", out.shape == (B, TOTAL, D), str(out.shape))

    exp = _np_expected(t, seg)
    check("与 numpy 参照逐位一致(选择+光栅序)",
          np.allclose(out, exp, atol=1e-5),
          f"max|Δ|={np.abs(out - exp).max():.2e}")

    # 用可辨识张量验证:通道0=帧id,通道1=帧内光栅号
    tid = np.zeros((B, n, C, D), np.float32)
    fid = np.arange(n)[None, :, None]
    rid = np.arange(C)[None, None, :]
    tid[..., 0] = np.broadcast_to(fid, (B, n, C))
    tid[..., 1] = np.broadcast_to(rid, (B, n, C))
    # 让分数可控:通道2 随机 → 打分随机但每帧仍恰选 seg 个
    tid[..., 2] = rng.standard_normal((B, n, C))
    o2 = np.asarray(_compress_soft_tokens(
        jnp.asarray(tid), 0, "dynseg", seg=jnp.asarray(seg), total=TOTAL))
    fr = o2[..., 0].astype(int)                       # [B,TOTAL] 帧id
    # 帧id 非递减(帧序保持)
    check("输出按帧序(帧id非递减)",
          all((np.diff(fr[b]) >= 0).all() for b in range(B)))
    # 逐帧贡献数 == seg
    okc = True
    for b in range(B):
        bc = np.bincount(fr[b], minlength=n)
        okc = okc and (bc == seg[b]).all()
    check("逐帧贡献 token 数 == seg(预算大帧贡献多)", okc,
          f"b0 counts={np.bincount(fr[0], minlength=n).tolist()}")
    # 帧内光栅序保持(同帧内通道1升序)
    okr = True
    for b in range(B):
        for i in range(n):
            m = fr[b] == i
            rs = o2[b, m, 1].astype(int)
            okr = okr and (np.diff(rs) > 0).all() if m.sum() > 1 else okr
    check("帧内光栅序保持(同帧内光栅号升序)", okr)

    # FRAME_SUBSAMPLE 情形:n*C == TOTAL(每帧满 64)
    n3 = 8
    t3 = rng.standard_normal((1, n3, C, D)).astype(np.float32)
    seg3 = np.full((1, n3), 64, np.int32)
    o3 = np.asarray(_compress_soft_tokens(
        jnp.asarray(t3), 0, "dynseg", seg=jnp.asarray(seg3), total=n3 * C))
    check("n*C==TOTAL 边界:形状正确", o3.shape == (1, n3 * C, D), str(o3.shape))


# --------------------------- (c) 逐样本模板 ------------------------------
def _runs(seq, head, delim, tail):
    """从建好的序列解析:去 head/tail 后按 delim 切,返回各段哨兵游程长度。"""
    body = seq[len(head):len(seq) - len(tail)]
    segs, cur, i = [], 0, 0
    while i < len(body):
        if delim and body[i:i + len(delim)] == list(delim):
            segs.append(cur)
            cur = 0
            i += len(delim)
        else:
            assert body[i] == SENTINEL, f"非哨兵/delim: {body[i]}"
            cur += 1
            i += 1
    segs.append(cur)
    return segs


def test_template():
    print("(c) _dynseg_template")
    head = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]          # 11 tok(仿 hf_layout)
    delim = [258882, 236743, 236771, 236771, 236787, 236771,
             236771, 236743, 255999]                    # 9 tok
    tail = list(range(1000, 1000 + 189))                # 189 tok
    TOTAL, FLOOR, n = 512, 8, 16
    Ts = []
    for seed in range(5):
        c = _dynseg_budget([_texture(seed * 17 + i) for i in range(n)],
                           TOTAL, FLOOR)
        seq = _dynseg_template(head, delim, c, tail)
        Ts.append(len(seq))
        n_sent = sum(1 for x in seq if x == SENTINEL)
        n_delim = sum(1 for j in range(len(seq) - len(delim) + 1)
                      if seq[j:j + len(delim)] == delim)
        runs = _runs(seq, head, delim, tail)
        check(f"[seed{seed}] 哨兵总数==TOTAL", n_sent == TOTAL, f"{n_sent}")
        check(f"[seed{seed}] delim 数==n-1", n_delim == n - 1, f"{n_delim}")
        check(f"[seed{seed}] 逐帧游程==counts",
              runs == c.tolist(), f"runs={runs[:4]}... c={c[:4].tolist()}...")
    T0 = 11 + TOTAL + (n - 1) * 9 + 189
    check("T 跨样本恒定", len(set(Ts)) == 1 and Ts[0] == T0,
          f"Ts={Ts} 期望={T0}")
    # FRAME_SUBSAMPLE:n=8 时 T 仍恒定(另一常量)
    n2 = 8
    c2 = _dynseg_budget([_texture(i) for i in range(n2)], TOTAL, FLOOR)
    seq2 = _dynseg_template(head, delim, c2, tail)
    check("n=8 时 T 恒定", len(seq2) == 11 + TOTAL + (n2 - 1) * 9 + 189,
          f"T={len(seq2)}")


def main():
    print("=" * 60)
    print("dynseg CPU 单测(JAX_PLATFORMS=%s)" % os.environ.get("JAX_PLATFORMS"))
    print("=" * 60)
    test_budget()
    test_device()
    test_template()
    print("=" * 60)
    if _FAILS:
        print(f"结果: FAIL —— {len(_FAILS)} 项未过: {_FAILS}")
        raise SystemExit(1)
    print("结果: ALL PASS")


if __name__ == "__main__":
    main()
