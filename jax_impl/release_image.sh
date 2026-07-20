#!/usr/bin/env bash
# 环境镜像发布 —— 仅当**依赖**变化时使用(jax/gemma/pip 包/基础镜像)。
#
#   sudo bash jax_impl/release_image.sh env-v2
#
# v1.8 之后代码与镜像分离: 代码改动只需 git push,客户 git pull + 重启
# 容器即生效,不发新镜像。本脚本只在环境变化时出场,tag 用 env-vN 语义。
# (注意用 sudo 而非 sudo -E: -E 保留 HOME 会让 docker 读错凭据文件)
set -e
cd "$(dirname "$0")/.."
VER=${1:?用法: release_image.sh <环境版本,如 env-v2>}
case "$VER" in env-v*) ;; *) echo "❌ 环境 tag 须为 env-vN 形式"; exit 1;; esac
AR=europe-west4-docker.pkg.dev/leeway-main/anker/jax

if [ -n "$(git status --porcelain jax_impl/Dockerfile)" ]; then
  echo "❌ Dockerfile 未提交 —— 环境定义必须先进 git"; exit 1
fi
SHA=$(git rev-parse --short HEAD)
echo "[release] 环境镜像 $AR:$VER(Dockerfile@$SHA)"

docker build -t "anker-jax:$VER" -f jax_impl/Dockerfile .

# 环境抽查: 依赖版本断言(构建期 selfcheck 已跑,这里防缓存伪影)
docker run --rm "anker-jax:$VER" python -c "
import jax; assert jax.__version__ == '0.10.2'
from gemma import gm; assert hasattr(gm.nn, 'Gemma4_E2B')
print('env content-check OK')"

docker tag "anker-jax:$VER" "$AR:$VER"
docker push "$AR:$VER"

echo
echo "✅ 已发布环境镜像 $AR:$VER"
echo "➡️  勿忘: 更新 jax_impl/USAGE.md 镜像引用,并通知所有使用者 docker pull"
