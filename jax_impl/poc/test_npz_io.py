"""npz_io 单测(纯 CPU,合成树): 三个致命坑修复的行为验证。
运行: JAX_PLATFORMS=cpu python jax_impl/poc/test_npz_io.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import jax.numpy as jnp

from jax_impl.npz_io import (detect_rank_scheme, load_lora_strict,
                             merge_proj_into_base, restore_train_tree)


def npz_of(d):
    f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(f.name, **d)
    return np.load(f.name)


def expect_raise(fn, frag):
    try:
        fn()
    except ValueError as e:
        assert frag in str(e), f"报错文案缺 {frag!r}: {e}"
        return
    raise AssertionError(f"应报错({frag})却成功")


# ---- detect_rank_scheme ----
z_prod = npz_of({"lora/layer_4/attn/q/lora/a": np.zeros((8, 512)),
                 "lora/layer_0/attn/q/lora/a": np.ones((8, 256)),
                 "lora/vision_encoder/x/lora/a": np.ones((8, 256))})
assert detect_rank_scheme(z_prod) == ("prod", [256, 512])
z_uni = npz_of({"lora/layer_0/attn/q/lora/a": np.ones((8, 16))})
assert detect_rank_scheme(z_uni) == ("uniform", [16])
expect_raise(lambda: detect_rank_scheme(npz_of({"proj/w": np.ones(3)})),
             "无 LLM LoRA")
print("OK detect_rank_scheme")

# ---- load_lora_strict ----
struct = {"layer_0": {"q": {"a": jnp.zeros((8, 256)),
                            "b": jnp.zeros((256, 8))}}}
good = npz_of({"lora/layer_0/q/a": np.ones((8, 256)),
               "lora/layer_0/q/b": np.zeros((256, 8))})
out = load_lora_strict(good, struct, jnp, jnp.float32)
assert float(out["layer_0"]["q"]["a"].sum()) == 8 * 256
# 形状不符(prod 产物 × uniform 结构)必须硬报错,不得静默填零
bad_shape = npz_of({"lora/layer_0/q/a": np.ones((8, 512)),
                    "lora/layer_0/q/b": np.zeros((512, 8))})
expect_raise(lambda: load_lora_strict(bad_shape, struct, jnp, jnp.float32),
             "未命中")
# 键缺失同理
expect_raise(lambda: load_lora_strict(
    npz_of({"lora/layer_0/q/a": np.ones((8, 256))}), struct, jnp,
    jnp.float32), "未命中")
# 全零 LoRA(stage a 产物)不得用于推理/KTO
all_zero = npz_of({"lora/layer_0/q/a": np.zeros((8, 256)),
                   "lora/layer_0/q/b": np.zeros((256, 8))})
expect_raise(lambda: load_lora_strict(all_zero, struct, jnp, jnp.float32),
             "全零")
print("OK load_lora_strict")

# ---- merge_proj_into_base ----
base = {"embedder": {"mm_input_projection": {"w": jnp.zeros((4, 6))},
                     "other": jnp.ones((2,))},
        "layer_0": {"w": jnp.ones((3,))}}
zp = npz_of({"proj/mm_input_projection/w": np.ones((4, 6)),
             "lora/layer_0/q/a": np.ones((8, 256))})
merged, n = merge_proj_into_base(base, zp, jnp, jnp.float32, required=True)
assert n == 1 and float(merged["embedder"]["mm_input_projection"]["w"].sum()) == 24
assert float(base["embedder"]["mm_input_projection"]["w"].sum()) == 0  # 原树不动
# npz 无 proj/: required=True 报错,False 放行
z_nop = npz_of({"lora/layer_0/q/a": np.ones((8, 256))})
expect_raise(lambda: merge_proj_into_base(base, z_nop, jnp, jnp.float32,
                                          required=True), "无 proj/")
_, n0 = merge_proj_into_base(base, z_nop, jnp, jnp.float32, required=False)
assert n0 == 0
# 形状不符报错
expect_raise(lambda: merge_proj_into_base(
    base, npz_of({"proj/mm_input_projection/w": np.ones((6, 4))}),
    jnp, jnp.float32, required=True), "形状不符")
print("OK merge_proj_into_base")

# ---- restore_train_tree(坑③)----
fresh_a = jnp.asarray(np.random.RandomState(0).normal(0, .02, (8, 256)),
                      jnp.float32)
train0 = {"lora": {"layer_0": {"q": {"a": fresh_a,
                                     "b": jnp.zeros((256, 8))}}},
          "proj": {"mm_input_projection": {"w": jnp.zeros((4, 6))}}}
# 场景 A: stage a 产物(lora 全零 + proj 有值)同方案续训
stage_a = npz_of({"lora/layer_0/q/a": np.zeros((8, 256)),
                  "lora/layer_0/q/b": np.zeros((256, 8)),
                  "proj/mm_input_projection/w": np.ones((4, 6))})
out, st = restore_train_tree(train0, stage_a, jnp,
                             is_zero_skippable=lambda k: "/layer_" in k)
assert st["zero_a_skip"] == 1, st
assert np.allclose(out["lora"]["layer_0"]["q"]["a"], fresh_a)  # 新初始化保留
assert float(out["proj"]["mm_input_projection"]["w"].sum()) == 24  # proj 恢复
# 场景 B: 真续训(a 非零)正常恢复
trained = npz_of({"lora/layer_0/q/a": np.full((8, 256), .5),
                  "lora/layer_0/q/b": np.full((256, 8), .1),
                  "proj/mm_input_projection/w": np.ones((4, 6))})
out, st = restore_train_tree(train0, trained, jnp,
                             is_zero_skippable=lambda k: "/layer_" in k)
assert st["zero_a_skip"] == 0 and st["hit"] == 3
assert float(out["lora"]["layer_0"]["q"]["a"][0, 0]) == .5
# 场景 C: 跨方案(形状不符)按预期跳过并计数
cross = npz_of({"lora/layer_0/q/a": np.full((8, 512), .5),
                "proj/mm_input_projection/w": np.ones((4, 6))})
out, st = restore_train_tree(train0, cross, jnp,
                             is_zero_skippable=lambda k: "/layer_" in k)
assert st["shape_skip"] == 1 and st["hit"] == 1
print("OK restore_train_tree")
print("ALL PASS")
