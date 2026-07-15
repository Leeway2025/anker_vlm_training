"""Gate A: JAX 侧加载 Gemma 4 E2B —— 类可用 / 权重可下 / 视觉模块存在。

用法(CPU 模式不抢 TPU;真机验证时把 JAX_PLATFORMS 去掉):
  JAX_PLATFORMS=cpu /dev/shm/venv_jax/bin/python jax_impl/poc/01_load_model.py

判定:
  PASS = 三段全部打印 [OK](类实例化 / checkpoint 加载 / 视觉参数树非空)
  权重下载失败多半是 gs:// 权限 —— 打印出的 path 拿去 gsutil ls 排查。
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "cpu"))


def main():
    import jax
    from gemma import gm
    print(f"[env] jax={jax.__version__} devices={jax.devices()}")

    # ---- 1. 模型类 ----
    cands = [n for n in dir(gm.nn) if "gemma4" in n.lower() or "e2b" in n.lower()]
    print(f"[introspect] gm.nn 中的 Gemma4/E2B 候选: {cands}")
    if not hasattr(gm.nn, "Gemma4_E2B"):
        print("[FAIL] 没有 gm.nn.Gemma4_E2B —— 升级 gemma 库或按上面候选改名")
        sys.exit(1)
    model = gm.nn.Gemma4_E2B()
    print(f"[OK] 模型类实例化: {type(model).__name__}")

    # ---- 2. checkpoint ----
    cps = [n for n in dir(gm.ckpts.CheckpointPath) if "GEMMA4" in n]
    print(f"[introspect] GEMMA4 checkpoint 常量: {cps}")
    path = gm.ckpts.CheckpointPath.GEMMA4_E2B_IT
    print(f"[info] checkpoint path = {path!s}")
    params = gm.ckpts.load_params(path)
    n = sum(x.size for x in jax.tree.leaves(params))
    print(f"[OK] 权重加载: {n/1e9:.2f}B 参数(预期 ≈5B 原始参数)")

    # ---- 3. 视觉/音频参数树 ----
    top = sorted({str(k) for k in params})
    vis = [k for k in top if "vision" in k.lower() or "vit" in k.lower()]
    aud = [k for k in top if "audio" in k.lower()]
    print(f"[info] 顶层参数树({len(top)}): {top[:12]}")
    print(f"[{'OK' if vis else 'FAIL'}] 视觉参数树: {vis[:4]}")
    print(f"[info] 音频参数树(我们会剥离,仅确认结构): {aud[:2]}")
    print("\nGate A:", "PASS" if vis else "NO-GO(视觉权重缺失)")


if __name__ == "__main__":
    main()
