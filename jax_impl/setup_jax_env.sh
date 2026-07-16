#!/usr/bin/env bash
# JAX 路线独立环境 —— 与 torch 训练 venv 完全隔离,互不影响。
# 要求 python>=3.12(gemma 库硬性要求);机器只有 3.10 时用 uv 拉独立解释器:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   UV_CACHE_DIR=/dev/shm/uv_cache TMPDIR=/dev/shm/tmp \
#     uv venv --python 3.12 /dev/shm/venv_jax && 用 uv pip 装下面同一批包
# ⚠️ 小系统盘机器务必把 UV_CACHE_DIR/TMPDIR 指到大盘(依赖树含 tensorflow)。
set -e
VENV=${1:-/dev/shm/venv_jax}
# pin 到验证过的 commit: jax_impl 对 gemma 内部做了 monkeypatch
# (Block remat / remove_mm_logits / _encode_vision 批量 / prod LoRA),
# 升级 gemma 前必须重跑 jax_impl 验证电池(FINDINGS 有清单)
GEMMA_PIN="git+https://github.com/google-deepmind/gemma.git@09e7b48ae88720f6236b8266c7213eb51bb62b87"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q -U "jax[tpu]" "$GEMMA_PIN" \
  kagglehub pillow numpy safetensors \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"$VENV/bin/python" - <<'PY'
import jax, gemma
print("jax", jax.__version__, "| gemma", getattr(gemma, "__version__", "?"))
from gemma import gm
assert hasattr(gm.nn, "Gemma4_E2B")
print("Gemma4_E2B OK")
PY
echo "OK -> $VENV"
