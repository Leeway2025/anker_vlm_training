# JAX 训练路线 **环境镜像**(纯环境件,不含代码)。
#
# 版本口径(v1.8 之后代码与镜像分离):
#   - 代码版本 = git commit(运行时经 -v $PWD:/workspace 挂载)
#   - 环境版本 = 本镜像 tag(env-vN),仅依赖变化时发新版
# 运行(TPU VM):
#   docker run --rm --privileged --net=host \
#     --ulimit nofile=1048576:1048576 --ulimit memlock=-1 \
#     -v /dev:/dev -v $PWD:/workspace -v /path/to/DATA:/data -w /workspace \
#     <镜像>:env-v1 python jax_impl/train_sft.py ...
# 忘挂 /workspace 会直接报"找不到 jax_impl/..."(响亮失败,防静默跑错版本)。
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

# 环境自检(构建期即发现环境坏): 只查依赖,不查代码
RUN python -c "import jax, gemma, PIL, numpy, safetensors; \
    from gemma import gm; \
    assert jax.__version__ == '0.10.2', jax.__version__; \
    assert hasattr(gm.nn, 'Gemma4_E2B'); \
    print('env selfcheck OK: jax', jax.__version__)"

WORKDIR /workspace
CMD ["python", "-c", "print('anker-jax env image; 代码请挂载 -v $PWD:/workspace,见 jax_impl/USAGE.md')"]
