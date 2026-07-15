"""JAX SFT 训练循环 v1(LoRA-only,单设备,验证机制与吞吐)。

  /dev/shm/venv_jax/bin/python jax_impl/train_sft.py \
      --labels /dev/shm/fakedata/labels.jsonl \
      --layout /dev/shm/hf_layout.json --steps 10 --accum 4 --lr 1e-4

设计(与 torch 路线的差异见 jax_impl/FINDINGS.md):
  - gm.nn.LoRA 整模包装(LLM einsum + 视觉塔全部注入;base 冻结,
    只训 lora 参数 —— 天然满足 RK1828 base 不动的红线)
  - 加权 CE: 分类 token ×4 + label_smoothing 0.1(对齐 torch loss)
  - 梯度累积: optax.MultiSteps(设备上累积,无 host 同步)
  - v1 无 remat(JAX 显存管理 + bs1 直测;不足再加 selective remat)
  - checkpoint: lora 参数树 npz;→ HF peft 导出走 export_hf.py(Gate D 链)
"""
import argparse
import dataclasses
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--out", default="/dev/shm/out_jax_sft")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from gemma import gm
    from gemma import peft as gpeft
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput

    from jax_impl.data import SftDataset, make_vision_input

    print(f"[env] devices={jax.devices()}")

    # ---- 逐层重算: gm 无内置 remat,动态把 Block 包成 nn.remat ----
    # (只影响本进程,不改库;policy 可换 dots_saveable 做选择性重算)
    import flax.linen as nn
    from gemma.gm.nn.gemma4 import _modules as g4_modules
    if not getattr(g4_modules, "_REMAT_PATCHED", False):
        # 整 Block 逐层重算。skip_sliding_mask 是 python bool,直接进
        # nn.remat 会被提升成 traced array 报错 → 在 remat 边界外按其值
        # 选择两个"值已固化在闭包里"的 remat 核,布尔不再进入 trace。
        _orig_call = g4_modules.Block.__call__
        _POL = jax.checkpoint_policies.nothing_saveable

        def _make_core(skip_flag):
            def core(self, x, pos, cache, mask, pli, kvs):
                return _orig_call(self, x, pos, cache, mask, pli, kvs,
                                  skip_sliding_mask=skip_flag)
            return nn.remat(core, policy=_POL, prevent_cse=False)

        _core = {False: _make_core(False), True: _make_core(True)}

        def patched(self, x, segment_pos, cache, attn_mask,
                    per_layer_input=None, kv_shared_cache=None,
                    skip_sliding_mask=False):
            return _core[bool(skip_sliding_mask)](
                self, x, segment_pos, cache, attn_mask,
                per_layer_input, kv_shared_cache)

        g4_modules.Block.__call__ = patched
        g4_modules._REMAT_PATCHED = True
        print("[remat] Block 已包装(闭包固化 bool, nothing_saveable)")

    # HF 语义对齐(勿删): gm 训练路径会 remove_mm_logits 把视觉位 logits
    # 压缩移除(gm 生态假设模型自己插入软 token,输出对齐回未插入序列),
    # 压缩后尾部是垃圾值,正好盖住我们的 label 窗口(实测尾部 670 位
    # "NaN")。我们按 HF 语义自行预插入/自行对齐 labels → 恒等旁路。
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    g4_tr._token_utils.remove_mm_logits = \
        lambda logits, tokens, num_tokens_per_image: logits
    print("[patch] remove_mm_logits 旁路(保持 HF 位置对齐)")

    # ---- 模型(视频语义: output_length=64,Gate C 配方)----
    TEXT_ONLY = bool(os.environ.get("REPRO_TEXT_ONLY"))   # 显存消融开关
    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    inner = gm.nn.Gemma4_E2B() if TEXT_ONLY else \
        gm.nn.Gemma4_E2B(text_only=False, config=cfg64)
    model = gm.nn.LoRA(rank=a.rank, model=inner)
    if TEXT_ONLY:
        print("[ablate] TEXT_ONLY: 无视觉塔,哨兵→0,images=None")

    tok = gm.text.Gemma4Tokenizer()
    ds = SftDataset(a.labels, a.layout, tok)
    print(f"[data] {len(ds)} samples, max_len={ds.max_len}")

    # ---- 参数: base 冻结加载 + lora 结构初始化 ----
    ex = ds[0]
    patches, pos_xy, counts = make_vision_input([ex["frames"]])
    dummy_tokens = jnp.asarray(ex["tokens"][None])
    dummy_pvi = PreprocessedVisionInput(
        patches=jnp.asarray(patches), positions_xy=jnp.asarray(pos_xy),
        soft_token_counts=counts)

    t0 = time.time()
    # eval_shape: 只拿结构不分配内存(直接 init 会在 TPU 物化 5B fp32 爆 HBM)
    if TEXT_ONLY:
        struct = jax.eval_shape(lambda: model.init(
            jax.random.PRNGKey(0), tokens=jnp.maximum(dummy_tokens, 0)))
    else:
        struct = jax.eval_shape(lambda: model.init(
            jax.random.PRNGKey(0), tokens=dummy_tokens, images=dummy_pvi))
    _, lora_struct = gpeft.split_params(struct["params"])
    rng = np.random.RandomState(0)
    def init_leaf(path, leaf):
        name = "/".join(getattr(k, "key", str(k)) for k in path)
        if name.endswith("/a"):        # A 随机、B 置零(peft 同款约定)
            return jnp.asarray(rng.normal(0, 0.02, leaf.shape), jnp.float32)
        return jnp.zeros(leaf.shape, jnp.float32)
    lora0 = jax.tree_util.tree_map_with_path(init_leaf, lora_struct)
    n_lora = sum(x.size for x in jax.tree.leaves(lora0))
    print(f"[init] lora params {n_lora/1e6:.1f}M ({time.time()-t0:.0f}s)")

    base = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    base = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), base)
    print(f"[init] base loaded ({time.time()-t0:.0f}s)")

    optim = optax.MultiSteps(optax.adamw(a.lr), every_k_schedule=a.accum)
    opt_state = optim.init(lora0)

    V = 262144
    ls = a.label_smoothing
    T = len(ds.template)               # label 只在尾窗 → 只算尾窗 CE

    def _is_trainable(path):
        # v1 与 torch adapter 同范围: 仅 LLM 层(attn+mlp)。视觉/embedder
        # LoRA 冻结 + stop_gradient —— 切断视觉塔反向(消融实测它吃 ~32G;
        # 视觉训练留给 v2: 给 vision/_modules 加同款 remat 后再放开)
        name = "/".join(getattr(k, "key", str(k)) for k in path)
        return name.startswith("layer_") or "/layer_" in name

    def loss_fn(lora, base_p, tokens, labels, weights, pvi):
        # LoRA 前向用 bf16(fp32 参与会把整条激活链提升成 fp32,HBM 翻倍);
        # fp32 master 权重仍在优化器侧(lora/opt_state 本体)
        lora_h = jax.tree_util.tree_map_with_path(
            lambda p, x: x.astype(jnp.bfloat16) if _is_trainable(p)
            else jax.lax.stop_gradient(x.astype(jnp.bfloat16)), lora)
        params = gpeft.merge_params(base_p, lora_h)
        if TEXT_ONLY:
            out = model.apply({"params": params},
                              tokens=jnp.maximum(tokens, 0))
        else:
            out = model.apply({"params": params}, tokens=tokens, images=pvi)
        logits = out.logits if hasattr(out, "logits") else out
        # shift + 尾窗: 位置 t 预测 t+1;label 区 [T, L)
        lg = logits[:, T - 1:-1].astype(jnp.float32)   # [B, K, V]
        lb = labels[:, T:]
        wt = weights[:, T:]
        valid = (lb != -100).astype(jnp.float32)
        lse = jax.nn.logsumexp(lg, axis=-1)             # [B, K]
        tgt = jnp.take_along_axis(
            lg, jnp.clip(lb, 0)[..., None], axis=-1)[..., 0]
        # 平滑 CE = (1-ls)(lse - lg_t) + ls·(lse - mean(lg)) —— 免 one-hot
        ce = (1 - ls) * (lse - tgt) + ls * (lse - lg.mean(-1))
        # PAD 位置的 logits 可为 NaN(全掩码行),而 NaN×0 仍是 NaN
        # → 必须先 where 归零再求和,不能靠 valid 乘法屏蔽
        ce = jnp.where(valid > 0, ce, 0.0)
        num = (ce * wt).sum()
        den = jnp.clip((wt * valid).sum(), 1.0)
        return num / den

    @__import__("functools").partial(jax.jit, donate_argnums=(0, 1))
    def train_step(lora, opt_state, base_p, tokens, labels, weights, pvi):
        loss, grads = jax.value_and_grad(loss_fn)(
            lora, base_p, tokens, labels, weights, pvi)
        updates, opt_state = optim.update(grads, opt_state, lora)
        lora = optax.apply_updates(lora, updates)
        return lora, opt_state, loss

    lora = lora0
    os.makedirs(a.out, exist_ok=True)
    hist = []
    t0 = time.time()
    for step in range(a.steps * a.accum):        # steps = 优化器步
        exi = ds[step % len(ds)]
        patches, pos_xy, counts = make_vision_input([exi["frames"]])
        pvi = PreprocessedVisionInput(
            patches=jnp.asarray(patches), positions_xy=jnp.asarray(pos_xy),
            soft_token_counts=counts)
        lora, opt_state, loss = train_step(
            lora, opt_state, base,
            jnp.asarray(exi["tokens"][None]),
            jnp.asarray(exi["labels"][None]),
            jnp.asarray(exi["weights"][None]), pvi)
        if step == 0:
            print(f"[compile+step0] {time.time()-t0:.0f}s")
            t0 = time.time()
        if (step + 1) % a.accum == 0:
            l = float(loss)
            hist.append(l)
            opt_step = (step + 1) // a.accum
            dt = (time.time() - t0) / max(opt_step * a.accum - 1, 1)
            print(f"[sft] opt_step {opt_step}/{a.steps} loss={l:.4f} "
                  f"micro_s/it={dt:.2f}", flush=True)
    try:
        ms = jax.local_devices()[0].memory_stats()
        print(f"[hbm] peak={ms.get('peak_bytes_in_use', 0)/2**30:.2f}G "
              f"limit={ms.get('bytes_limit', 0)/2**30:.2f}G")
    except Exception:  # noqa: BLE001
        pass

    flat = jax.tree_util.tree_flatten_with_path(lora)[0]
    np.savez(os.path.join(a.out, "lora_params.npz"),
             **{"/".join(getattr(k, "key", str(k)) for k in p):
                np.asarray(v) for p, v in flat})
    json.dump({"loss_history": hist, "rank": a.rank},
              open(os.path.join(a.out, "train_meta.json"), "w"))
    print(f"[save] {a.out} (loss {hist[0]:.3f} -> {hist[-1]:.3f})")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
