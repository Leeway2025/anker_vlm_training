# JAX 训练路线容器 —— 团队环境对齐用(与 setup_jax_env.sh 同源,全 pin)。
#
# 构建(仓库根目录):
#   docker build -t anker-jax:v1 -f jax_impl/Dockerfile .
# 运行(TPU VM;--privileged + /dev 挂载让容器看到 TPU 芯):
#   docker run --rm --privileged --net=host \
#     -v /dev:/dev -v $PWD:/workspace -v /path/to/DATA:/data -w /workspace \
#     anker-jax:v1 python jax_impl/train_sft.py \
#       --labels /data/labels.jsonl --layout /data/hf_layout.json ...
# 说明:
#   - 镜像已烘入本仓库代码(/app);挂 -w /workspace 用宿主机最新代码,
#     不挂则用镜像内固化版本(严格对齐场景);
#   - GCS 凭据经 --net=host 走 VM metadata(gs://gemma-data 权重直读);
#   - 非标准 GCE 机器(ct6e 等)须额外传 TPU_* 环境变量,见 FINDINGS。
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PJRT_DEVICE=TPU

# 版本 pin 与验证电池一致: jax 0.10.2 / gemma@09e7b48(勿动;
# 升级任一项都必须重跑 jax_impl 验证电池,见 FINDINGS)
RUN pip install \
      "jax[tpu]==0.10.2" \
      -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
      "git+https://github.com/google-deepmind/gemma.git@09e7b48ae88720f6236b8266c7213eb51bb62b87" \
      kagglehub pillow numpy safetensors

WORKDIR /app
# jax_impl 及其依赖的纯 python 模块(已核验零 torch 依赖)
COPY jax_impl/ /app/jax_impl/
COPY data/__init__.py data/taxonomy.py data/formatting.py /app/data/
COPY configs/ /app/configs/

# 自检: 关键导入与 Gemma4 类存在(构建期即发现环境坏)
RUN python -c "import jax, gemma; from gemma import gm; \
    assert hasattr(gm.nn, 'Gemma4_E2B'); \
    import sys; sys.path.insert(0, '/app'); \
    import jax_impl.prod_lora, jax_impl.prefetch; \
    print('image selfcheck OK: jax', jax.__version__)"

ENV PYTHONPATH=/app
CMD ["python", "-c", "print('anker-jax image ready; see jax_impl/USAGE.md')"]
