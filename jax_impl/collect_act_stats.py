"""LoRA 注入点激活统计采集(激活感知 SVD 截断的前置,TPU 上跑)。

  python3 jax_impl/collect_act_stats.py --ckpt outputs/jax_5b_s4/train_params_best.npz \
      --labels DATA/labels_dedup.jsonl --layout DATA/hf_layout.json \
      [--wds-dir ...] [--n 256] --out outputs/act_stats.npz

原理: 每个 LoRA 适配器前向时,把输入 x 的逐元素平方按 a 的非 rank 维
归约(与 svd_truncate_lora 的 In 展平严格同序),io_callback 累加到
host;跑 N 条真实样本后取均值 E[x²],即 ASVD 白化对角。

产出 npz: {<npz lora 键去掉 lora/ 前缀与 /a 后缀>: E[x²] 向量}
—— svd_truncate_lora.py --act-stats 直接消费,键已对齐(采集完就地
用 --ckpt 的 lora 键做过校验/重映射,缺失会响亮告警)。

骨架与 infer.py 同源(prod patch/严格加载/proj 上车);单机单进程,
N=256 约几分钟。⚠️ 待真机门禁: 首次使用先跑 --n 8 冒烟,确认
[act] 命中 = lora 对数。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATS = {}          # key → [sum_xsq(np), count(float)]


def _install_stats_tap():
    """在 prod 前向上追加统计旁路(不改变输出)。"""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from gemma.peft import _lora

    orig_ein = _lora.LoRAEinsumAdapter.__call__
    orig_dense = _lora.LoRADenseAdapter.__call__

    def _acc(key, shape):
        def cb(xsq):
            ent = STATS.setdefault(key, [np.zeros(xsq.shape, np.float64), 0.0])
            ent[0] += np.asarray(xsq, np.float64)
            ent[1] += 1.0
        return cb

    def ein_call(self, inputs):
        # 'inputs,a_str,b_str->out' → 归约到 a 的非 rank 维(与截断同序)
        es = self._lora_einsum_str
        in_str, a_str = es.split("->")[0].split(",")[:2]
        red = f"{in_str}->{a_str[:-1]}"
        xsq = jnp.einsum(red, (inputs * inputs).astype(jnp.float32))
        n = 1.0
        for i, c in enumerate(in_str):
            if c not in a_str[:-1]:
                n *= inputs.shape[i]
        key = "/".join(str(s) for s in (self.scope.path or ()))
        jax.experimental.io_callback(_acc(key, None), None, xsq / max(n, 1.0),
                                     ordered=False)
        return orig_ein(self, inputs)

    def dense_call(self, inputs):
        xsq = (inputs * inputs).astype(jnp.float32)
        xsq = xsq.reshape(-1, inputs.shape[-1]).mean(0)
        key = "/".join(str(s) for s in (self.scope.path or ()))
        jax.experimental.io_callback(_acc(key, None), None, xsq,
                                     ordered=False)
        return orig_dense(self, inputs)

    _lora.LoRAEinsumAdapter.__call__ = ein_call
    _lora.LoRADenseAdapter.__call__ = dense_call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="prod 训练产物 npz")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--wds-dir", default=None)
    ap.add_argument("--n", type=int, default=256, help="采样条数")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import dataclasses
    import numpy as np
    import jax
    import jax.numpy as jnp
    from gemma import gm
    from gemma import peft as gpeft
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
    from jax_impl.data import SftDataset, make_vision_input
    from jax_impl.npz_io import (detect_rank_scheme, load_lora_strict,
                                 merge_proj_into_base)
    from jax_impl.prod_lora import install_prod_lora

    z = np.load(a.ckpt)
    scheme, ranks = detect_rank_scheme(z)
    print(f"[scheme] {scheme} ranks={ranks}")
    if scheme == "prod":
        install_prod_lora()
    _install_stats_tap()                      # 必须在模型构造前
    # infer 同款恒等旁路(漏打 → 尾窗错位)
    g4_tr._token_utils.remove_mm_logits = \
        lambda logits, tokens, num_tokens_per_image: logits

    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    tok = gm.text.Gemma4Tokenizer()
    ds = SftDataset(a.labels, a.layout, tok, wds_dir=a.wds_dir)
    T = len(ds.template)
    model = gm.nn.LoRA(rank=ranks[0],
                       model=gm.nn.Gemma4_E2B(text_only=False, config=cfg64))
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    params = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), params)
    params, n_proj = merge_proj_into_base(params, z, jnp, jnp.bfloat16,
                                          required=False)
    ex0 = ds[0]
    p0, x0, counts0 = make_vision_input([ex0["frames"]])
    L = T + 4
    struct = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0), tokens=jnp.zeros((1, L), jnp.int32),
        images=PreprocessedVisionInput(
            patches=jnp.asarray(p0), positions_xy=jnp.asarray(x0),
            soft_token_counts=counts0)))
    lora_struct = gpeft.split_params(struct["params"])[1]
    lora = load_lora_strict(z, lora_struct, jnp, jnp.bfloat16)
    params = gpeft.merge_params(params, lora)
    print(f"[init] lora+proj({n_proj}叶) from {a.ckpt}")

    @jax.jit
    def fwd(par, tokens, pvi):
        out = model.apply({"params": par}, tokens=tokens, images=pvi)
        return (out.logits if hasattr(out, "logits") else out).sum()

    idxs = list(range(min(a.n, len(ds.train_idx))))
    for i, di in enumerate(idxs):
        ex = ds[ds.train_idx[di]]
        pt, px, cnt = make_vision_input([ex["frames"]])
        pvi = PreprocessedVisionInput(
            patches=jnp.asarray(pt), positions_xy=jnp.asarray(px),
            soft_token_counts=cnt)
        toks = np.zeros((1, L), np.int32)
        toks[0, :T] = ex["tokens"][:T]
        fwd(params, jnp.asarray(toks), pvi).block_until_ready()
        if (i + 1) % 32 == 0:
            print(f"  {i+1}/{len(idxs)}")

    # ---- 键对齐校验: STATS 键须与 npz 'lora/<key>/a' 一一对上 ----
    want = {k[len("lora/"):-len("/a")] for k in z.files
            if k.startswith("lora/") and k.endswith("/a")}
    out_map, miss = {}, []
    for wk in sorted(want):
        if wk in STATS:
            s, c = STATS[wk]
        else:                                  # 后缀模糊匹配(作用域前缀差异)
            cand = [k for k in STATS if k.endswith(wk) or wk.endswith(k)]
            if len(cand) == 1:
                s, c = STATS[cand[0]]
            else:
                miss.append(wk)
                continue
        out_map[wk] = (s / max(c, 1.0)).astype(np.float32)
    print(f"[act] 命中 {len(out_map)}/{len(want)} 对"
          + (f";⚠️ 缺失示例 {miss[:3]}(采集键样例 "
             f"{list(STATS)[:2]})" if miss else ""))
    if not out_map:
        raise SystemExit("全部未命中: 作用域键与 npz 键约定不一致,"
                         "把上面的样例键发回排查")
    np.savez(a.out, **out_map)
    print(f"[OK] E[x²] 统计({len(out_map)} 键, n={a.n})→ {a.out};"
          f"用法: svd_truncate_lora.py --act-stats {a.out}")


if __name__ == "__main__":
    main()
