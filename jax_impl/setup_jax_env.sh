#!/usr/bin/env bash
# JAX 路线独立环境 —— 与 torch 训练 venv 完全隔离,互不影响
set -e
VENV=${1:-/dev/shm/venv_jax}
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip
# jax[tpu]: libtpu 随 jax 版本锁定;gemma: google-deepmind 官方 JAX 库
"$VENV/bin/pip" install -q -U "jax[tpu]" gemma kagglehub pillow numpy \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"$VENV/bin/python" - <<'PY'
import jax, gemma
print("jax", jax.__version__, "| gemma", getattr(gemma, "__version__", "?"))
PY
echo "OK -> $VENV"
