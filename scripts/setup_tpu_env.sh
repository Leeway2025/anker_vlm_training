#!/bin/bash
# TPU VM 一键环境(版本 = 2026-07 v6e-1 全链验证通过的组合,勿随意升级)
# 用法: bash scripts/setup_tpu_env.sh [--with-model]
set -e

echo "== pip 栈(torch 与 torch_xla 版本必须一致,ABI 锁定)=="
pip install --quiet torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install --quiet torch_xla[tpu]==2.9.0
pip install --quiet torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
pip install --quiet "transformers>=4.53" "peft>=0.13" safetensors pyyaml \
    opencv-python-headless pillow sentencepiece accelerate tensorboard \
    "jinja2>=3.1" google-genai google-cloud-storage

echo "== 自检 =="
python3 - <<'EOF'
import torch, torch_xla
assert torch.__version__.startswith("2.9.0"), torch.__version__
assert torch_xla.__version__.startswith("2.9.0"), torch_xla.__version__
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
d = xm.xla_device()
x = torch.randn(256, 256, device=d) @ torch.randn(256, 256, device=d)
xm.mark_step()
print(f"TPU OK: device={d}, world={xr.world_size()}, "
      f"local devices={xr.global_runtime_device_count()}")
EOF
python3 tests/test_core.py | tail -1

if [[ "$1" == "--with-model" ]]; then
  echo "== 预下载 gemma-4-e2b-it(~10GB)=="
  python3 -c "from huggingface_hub import snapshot_download; \
      print(snapshot_download('google/gemma-4-e2b-it'))"
fi
echo "环境就绪。8 卡首跑清单见 docs/WALKTHROUGH.md"
