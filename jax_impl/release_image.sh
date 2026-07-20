#!/usr/bin/env bash
# 镜像发布(唯一入口)—— 强制"代码=镜像=文档"三方同步。
#
#   bash jax_impl/release_image.sh v1.2
#
# 铁律(v1 事故的教训: 构建时机早于修复,镜像带已知 bug 上线):
#   1. 工作区必须 clean —— 镜像内容严格等于当前 commit;
#   2. 双 tag: 语义版本 + git short-sha,不可变、不复用、永不用 latest;
#   3. 构建后抽查关键文件在镜像内(防 COPY 漏挂);
#   4. 结束打印 USAGE.md 待更新提醒(引用版本号)。
set -e
cd "$(dirname "$0")/.."
VER=${1:?用法: release_image.sh <语义版本,如 v1.2>}
AR=europe-west4-docker.pkg.dev/leeway-main/anker/jax

if [ -n "$(git status --porcelain)" ]; then
  echo "❌ 工作区不 clean —— 先提交/清理,保证镜像内容≡commit"; exit 1
fi
SHA=$(git rev-parse --short HEAD)
echo "[release] commit=$SHA -> $AR:{$VER,$SHA}"

docker build -t "anker-jax:$VER" -f jax_impl/Dockerfile .

# 内容抽查: 近期修复的关键标记必须在镜像内(按需追加)
docker run --rm "anker-jax:$VER" sh -c '
  set -e
  grep -q base_p /app/jax_impl/kto.py
  grep -q TPU_PROCESS_ADDRESSES /app/jax_impl/infer_sharded.sh
  grep -q _disable_ktyping /app/jax_impl/prefetch.py
  grep -q load_lora_strict /app/jax_impl/npz_io.py
  grep -q rank-scheme /app/jax_impl/infer.py
  grep -q split_by_camera /app/jax_impl/data.py
  echo content-check OK'

docker tag "anker-jax:$VER" "$AR:$VER"
docker tag "anker-jax:$VER" "$AR:$SHA"
docker push "$AR:$VER"
docker push "$AR:$SHA"

echo
echo "✅ 已发布 $AR:$VER(= commit $SHA)"
echo "➡️  勿忘: 更新 jax_impl/USAGE.md 中的镜像版本引用为 $VER 并提交"
grep -n "anker/jax:v" jax_impl/USAGE.md | head -3
