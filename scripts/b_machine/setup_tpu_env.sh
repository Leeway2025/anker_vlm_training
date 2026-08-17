#!/bin/bash
# TPU VM 一键环境(版本 = 2026-07 v6e-1 全链验证通过的组合,勿随意升级)
# 用法: bash scripts/setup_tpu_env.sh [--with-model]
set -e

# ── 非标准 TPU 环境(GKE 节点/自定义镜像)排障 ──────────────────
# 症状: torch_xla 启动即 KeyError: 'ACCELERATOR_TYPE'(tpu.py version())
# 原因: 该环境的 metadata `tpu-env` 键带 TPU_ 前缀,与 torch_xla 路径 A
#       的无前缀假设不符。解法 = 走 torch_xla 官方路径 B(环境变量),
#       两个变量必须成对(①绕开 metadata ②补上绕开后缺的数据):
# export TPU_SKIP_MDS_QUERY=1
# export TPU_ACCELERATOR_TYPE=v6e-8        # 按实际机型
# 若再报 bounds/worker None(单机 v6e-8):
# export TPU_PROCESS_BOUNDS=1,1,1
# export TPU_CHIPS_PER_PROCESS_BOUNDS=2,4,1
# export TPU_WORKER_ID=0
# 标准 Cloud TPU VM 不需要以上任何变量(metadata 即无前缀格式)。


echo "== pip 栈(torch 与 torch_xla 版本必须一致,ABI 锁定)=="
pip install --quiet torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install --quiet torch_xla[tpu]==2.9.0
pip install --quiet torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
# 2026-07-13 教训: 曾用浮动版本(transformers>=4.53),导致验证机与
# 客户机版本不可追溯;sentencepiece 曾靠间接依赖带入,客户机上缺失。
# 现全部精确 pin,与 requirements-lock.txt(验证机 freeze)一致:
pip install --quiet transformers==5.13.0 peft==0.19.1 accelerate==1.14.0 \
    safetensors==0.8.0 sentencepiece==0.2.1 pyyaml \
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
