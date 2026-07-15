"""JAX 批量推理(hard_mining / KTO 数据生成 / 评测用)。

  python jax_impl/infer.py --labels labels.jsonl --layout hf_layout.json \
      --out preds.jsonl [--limit 8] [--max-new 40] [--init-npz lora.npz]

实现: 固定步数贪心解码,全程静态形状 —— 每步对 [T+max_new] padded 全长
做一次前向,读位置 T+i-1 的 logits argmax 写回。只编译一张图,逐步复用
(HF generate 在 XLA 上逐 token 重编译的教训,见 torch 侧 _XlaStepMarker)。
无 KV cache: 每步全长前向,~0.5s/步 @v6e 单芯;小规模挖掘/评测够用,
大规模推理用 torch 侧 static-cache 路径或后续接 gm Sampler。
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EOT = 106


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--init-npz", help="载入训练产物 lora(缺省纯 base)")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from gemma import gm
    from gemma import peft as gpeft
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
    from jax_impl.data import SftDataset, make_vision_input

    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    tok = gm.text.Gemma4Tokenizer()
    ds = SftDataset(a.labels, a.layout, tok, max_label_len=a.max_new)
    T = len(ds.template)
    L = T + a.max_new

    if a.init_npz:
        model = gm.nn.LoRA(rank=int(np.load(a.init_npz)[
            [k for k in np.load(a.init_npz).files if k.endswith("/a")][0]
        ].shape[-1]), model=gm.nn.Gemma4_E2B(text_only=False, config=cfg64))
    else:
        model = gm.nn.Gemma4_E2B(text_only=False, config=cfg64)
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    params = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), params)
    if a.init_npz:
        z = np.load(a.init_npz)
        ex0 = ds[0]
        p0, x0, counts0 = make_vision_input([ex0["frames"]])
        struct = jax.eval_shape(lambda: model.init(
            jax.random.PRNGKey(0), tokens=jnp.zeros((1, L), jnp.int32),
            images=PreprocessedVisionInput(
                patches=jnp.asarray(p0), positions_xy=jnp.asarray(x0),
                soft_token_counts=counts0)))
        lora_struct = gpeft.split_params(struct["params"])[1]

        def fill(path, leaf):
            k = "/".join(getattr(x, "key", str(x)) for x in path)
            k = "lora/" + k
            if k in z.files and z[k].shape == leaf.shape:
                return jnp.asarray(z[k], jnp.bfloat16)
            return jnp.zeros(leaf.shape, jnp.bfloat16)
        lora = jax.tree_util.tree_map_with_path(fill, lora_struct)
        params = gpeft.merge_params(params, lora)
        print(f"[init] lora from {a.init_npz}")

    @jax.jit
    def step_logits(par, tokens, pvi, pos):
        out = model.apply({"params": par}, tokens=tokens, images=pvi)
        lg = out.logits if hasattr(out, "logits") else out
        return jnp.argmax(lg[0, pos - 1])

    recs = ds.recs[: a.limit] if a.limit else ds.recs
    fout = open(a.out, "w", encoding="utf-8")
    import time
    for n, rec in enumerate(recs):
        ex = ds[n]
        patches, pos_xy, counts = make_vision_input([ex["frames"]])
        pvi = PreprocessedVisionInput(
            patches=jnp.asarray(patches), positions_xy=jnp.asarray(pos_xy),
            soft_token_counts=counts)
        toks = np.zeros(L, np.int32)
        toks[:T] = ds.template
        t0 = time.time()
        out_ids = []
        for i in range(a.max_new):
            nxt = int(step_logits(params, jnp.asarray(toks[None]),
                                  pvi, T + i))
            out_ids.append(nxt)
            toks[T + i] = nxt
            if nxt == EOT:
                break
        txt = tok.decode([t for t in out_ids if t != EOT])
        fout.write(json.dumps({"video_id": rec["video_id"],
                               "output": txt.strip()},
                              ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[infer] {n+1}/{len(recs)} {rec['video_id']} "
              f"({time.time()-t0:.1f}s): {txt[:60]!r}", flush=True)
    print(f"[OK] -> {a.out}")


if __name__ == "__main__":
    main()
