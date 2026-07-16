"""生产 LoRA 方案注入: 差异化 rank + rsLoRA 缩放(对齐 torch 生产配置)。

torch 侧生产配置(从真实 adapter 逆向确认,见 FINDINGS v3):
  - 全局注意力层(E2B 每 5 层一个: 4,9,14,19,24,29,34)r=512,其余 r=256;
  - rsLoRA: scaling = alpha / sqrt(r),alpha = 2r
    → r256 层 scale=32,r512 层 scale=2*sqrt(512)≈45.25;
  - 视觉塔 r=256。

实现: monkeypatch gemma.peft._lora 的 Adapter(参数创建处可拿到
self.scope.path → 按路径定 rank;输出乘 scale)。gm.nn.LoRA(rank=...) 的
入参 rank 因此被本方案覆盖(仅作占位)。

⚠️ 与 import_hf 的 legacy 容器方案(uniform 2×max_r + scale 折进 B、
前向无缩放)不兼容 —— torch checkpoint 续训走 legacy,JAX 原生训练
+ export 走本方案;两者不得混用(FINDINGS 有说明)。
"""
import math

E2B_GLOBAL_LAYERS = frozenset({4, 9, 14, 19, 24, 29, 34})


def install_prod_lora(r_global=512, r_sliding=256, r_vision=256,
                      global_layers=E2B_GLOBAL_LAYERS,
                      alpha_mult=2.0, rslora=True):
    import flax.linen as nn
    import jax.numpy as jnp
    from gemma.peft import _lora
    from gemma.peft import _einsum_utils

    if getattr(_lora, "_PROD_PATCHED", False):
        return

    def eff_rank(path):
        segs = list(path or ())
        joined = "/".join(str(s) for s in segs)
        for s in segs:
            if isinstance(s, str) and s.startswith("layer_"):
                i = int(s.split("_")[1])
                return r_global if i in global_layers else r_sliding
        if "vision_encoder" in joined:
            return r_vision
        return r_sliding                      # embedder 等(默认冻结)

    def scale_for(r):
        alpha = alpha_mult * r
        return alpha / (math.sqrt(r) if rslora else r)

    # ---- Einsum adapter(LLM 与视觉的全部注入点)----
    def ein_setup(self):
        r = eff_rank(self.scope.path if self.scope else ())
        out = _einsum_utils.get_lora_einsum_str_and_shapes(
            einsum_str=self.einsum_str, weights_shape=self.shape, rank=r)
        (lora_einsum_str, a_shape, b_shape) = out
        self._lora_einsum_str = lora_einsum_str
        self._a = self.param("a", self.a_init, a_shape, dtype=self.dtype)
        self._b = self.param("b", self.b_init, b_shape, dtype=self.dtype)
        self._prod_scale = scale_for(r)

    def ein_call(self, inputs):
        return self._prod_scale * jnp.einsum(
            self._lora_einsum_str, inputs, self._a, self._b)

    _lora.LoRAEinsumAdapter.setup = ein_setup
    _lora.LoRAEinsumAdapter.__call__ = ein_call

    # ---- Dense adapter(防视觉塔某些层走 Dense 包装)----
    def dense_call(self, inputs):
        r = eff_rank(self.scope.path if self.scope else ())
        a = self.param("a", self.a_init,
                       (inputs.shape[-1], r), dtype=self.dtype)
        b = self.param("b", self.b_init,
                       (r, self.features), dtype=self.dtype)
        return scale_for(r) * (inputs @ a @ b)

    _lora.LoRADenseAdapter.__call__ = nn.compact(dense_call)

    _lora._PROD_PATCHED = True
    print(f"[prod-lora] r_global={r_global}@layers{sorted(global_layers)} "
          f"r_sliding={r_sliding} r_vision={r_vision} "
          f"rsLoRA α={alpha_mult}r (scale {scale_for(r_sliding):.2f}/"
          f"{scale_for(r_global):.2f})")


def prod_adapter_config(base_model="google/gemma-4-e2b-it",
                        r_sliding=256, r_global=512,
                        global_layers=E2B_GLOBAL_LAYERS, alpha_mult=2.0):
    """导出用: 与 torch 生产 adapter 同款的 peft 配置(rank/alpha_pattern)。"""
    projs = ("q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj")
    rank_pattern, alpha_pattern = {}, {}
    for i in sorted(global_layers):
        for p in projs:
            key = rf".*\.layers\.{i}\..*\.{p}"
            rank_pattern[key] = r_global
            alpha_pattern[key] = int(alpha_mult * r_global)
    return {
        "peft_type": "LORA", "r": r_sliding,
        "lora_alpha": int(alpha_mult * r_sliding),
        "use_rslora": True, "lora_dropout": 0.0, "bias": "none",
        "rank_pattern": rank_pattern, "alpha_pattern": alpha_pattern,
        "target_modules": (r".*language_model\.layers\.\d+\."
                           r"(self_attn\.(q|k|v|o)_proj|"
                           r"mlp\.(gate|up|down)_proj)"),
        "task_type": None, "base_model_name_or_path": base_model,
    }
