"""视觉塔 LoRA 映射矩阵级单测(纯 numpy)。

对每类模块: 随机 a/b → map_vision_loras 出 A/B → 比较
  y_jax = einsum 语义前向增量(按 lora 收缩: 入轴·a → r → b·出轴)
  y_hf  = B @ (A @ x)(peft 语义,HF 融合排布)
HF 融合排布(h*hd+d / gate|up 分支 / in-out 转置)已由 base 权重逐位
证明(axis_proof 2026-07-20);本测试钉住映射代码与该排布的一致性。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from jax_impl.export_hf import map_vision_loras, split_flat

rng = np.random.RandomState(0)
Lyr, D, H, hd, F, r = 3, 768, 12, 64, 3072, 16   # 缩小的同构形状
P = "vision_encoder/transformer/stacked_layers/block/"
flat = {
    P+"attn/q_einsum/_LoRAEinsum_0/lora/a": rng.normal(0,1,(Lyr,D,r)),
    P+"attn/q_einsum/_LoRAEinsum_0/lora/b": rng.normal(0,1,(Lyr,r,H,hd)),
    P+"attn/kv_einsum/_LoRAEinsum_0/lora/a": rng.normal(0,1,(Lyr,D,r)),
    P+"attn/kv_einsum/_LoRAEinsum_0/lora/b": rng.normal(0,1,(Lyr,r,2,H,hd)),
    P+"attn/attn_vec_einsum/_LoRAEinsum_0/lora/a": rng.normal(0,1,(Lyr,H,hd,r)),
    P+"attn/attn_vec_einsum/_LoRAEinsum_0/lora/b": rng.normal(0,1,(Lyr,r,D)),
    P+"mlp/gating_einsum/_LoRAEinsum_0/lora/a": rng.normal(0,1,(Lyr,D,r)),
    P+"mlp/gating_einsum/_LoRAEinsum_0/lora/b": rng.normal(0,1,(Lyr,r,2,F)),
    P+"mlp/linear/_LoRAEinsum_0/lora/a": rng.normal(0,1,(Lyr,F,r)),
    P+"mlp/linear/_LoRAEinsum_0/lora/b": rng.normal(0,1,(Lyr,r,D)),
    "vision_encoder/entry/input_projection/_LoRAEinsum_0/lora/a":
        rng.normal(0,1,(D,r)),   # 应告警跳过
}
llm, vis = split_flat(flat)
assert not llm
sd = map_vision_loras(vis)
assert len(sd) == Lyr * 7 * 2, len(sd)
pre = "base_model.model.model.vision_tower.encoder.layers"

def hfy(L, mod, x):
    A = sd[f"{pre}.{L}.{mod}.linear.lora_A.weight"]
    B = sd[f"{pre}.{L}.{mod}.linear.lora_B.weight"]
    return B @ (A @ x)

def rel(a, b):     # A/B 被 cast 成 f32 → 比相对误差(轴序错会是 O(1))
    a = np.asarray(a).reshape(-1); b = np.asarray(b).reshape(-1)
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-9))

worst = 0.0
for L in range(Lyr):
    x = rng.normal(0, 1, D)
    xf = rng.normal(0, 1, F)
    g = lambda k: flat[P+k][L]
    # q: (D)·a(D,r)·b(r,H,hd) → HF 融合 [h*hd+d]
    yj = np.einsum("d,dr,rnh->nh", x, g("attn/q_einsum/_LoRAEinsum_0/lora/a"),
                   g("attn/q_einsum/_LoRAEinsum_0/lora/b")).reshape(-1)
    worst = max(worst, rel(yj, hfy(L, "self_attn.q_proj", x)))
    # kv → k/v
    for j, m in ((0, "self_attn.k_proj"), (1, "self_attn.v_proj")):
        yj = np.einsum("d,dr,rnh->nh", x,
                       g("attn/kv_einsum/_LoRAEinsum_0/lora/a"),
                       g("attn/kv_einsum/_LoRAEinsum_0/lora/b")[:, j]).reshape(-1)
        worst = max(worst, rel(yj, hfy(L, m, x)))
    # o: 入侧融合 (H,hd) → HF in [h*hd+d]
    xo = rng.normal(0, 1, (H, hd))
    yj = np.einsum("nh,nhr,rd->d", xo,
                   g("attn/attn_vec_einsum/_LoRAEinsum_0/lora/a"),
                   g("attn/attn_vec_einsum/_LoRAEinsum_0/lora/b"))
    worst = max(worst, rel(yj, hfy(L, "self_attn.o_proj", xo.reshape(-1))))
    # gating → gate/up
    for j, m in ((0, "mlp.gate_proj"), (1, "mlp.up_proj")):
        yj = np.einsum("d,dr,rf->f", x,
                       g("mlp/gating_einsum/_LoRAEinsum_0/lora/a"),
                       g("mlp/gating_einsum/_LoRAEinsum_0/lora/b")[:, j])
        worst = max(worst, rel(yj, hfy(L, m, x)))
    # down
    yj = np.einsum("f,fr,rd->d", xf,
                   g("mlp/linear/_LoRAEinsum_0/lora/a"),
                   g("mlp/linear/_LoRAEinsum_0/lora/b"))
    worst = max(worst, rel(yj, hfy(L, "mlp.down_proj", xf)))
print(f"matrix-check worst={worst:.2e}")
assert worst < 1e-5, worst
# config 覆盖检查: 全部导出键能被 target_modules 正则命中
import re
from jax_impl.prod_lora import prod_adapter_config
tm = re.compile(prod_adapter_config()["target_modules"])
mods = {k.rsplit(".lora_", 1)[0].removeprefix("base_model.model.")
        for k in sd}
miss = [m for m in mods if not tm.match(m)]
assert not miss, miss[:3]
print("config target_modules 全覆盖", len(mods), "模块")
print("ALL PASS")
