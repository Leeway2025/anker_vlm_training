"""fd 级日志落盘: stdout/stderr 同时写终端(docker logs)与 <out>/<name>。

为什么在 fd 层做而不是包 sys.stdout: libtpu / XLA 的 C++ 侧日志直接写
文件描述符 2,python 层的 tee 根本看不到 —— 而 TPU 初始化失败
(DEADLINE_EXCEEDED / vfio busy)恰恰全是这类输出。现场排障要的就是它们。

用法(各训练/推理入口在 parse_args 后第一行调用):
    from jax_impl.logtee import tee_stdio
    tee_stdio(a.out)          # → <out>/train.log,追加模式
容器场景 <out> 落在 -v 挂载目录下即自动持久化到宿主机;容器 rm 不丢。
"""
import os
import subprocess
import sys

_KEEP = []          # 防 tee 进程与管道被 GC


def tee_stdio(out_dir, name="train.log"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    # tee 先于 dup2 启动 → 它继承的 stdout 仍是原终端(docker logs 不受影响)
    p = subprocess.Popen(["tee", "-a", path], stdin=subprocess.PIPE)
    _KEEP.append(p)
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(p.stdin.fileno(), 1)
    os.dup2(p.stdin.fileno(), 2)

    # 退出排空: 容器内 python 是 PID 1,直接退出会连带杀 tee,管道尾部
    # (往往正是 traceback)丢失 —— 实测 5 行截断。EOF + wait 保证写完。
    import atexit

    def _drain():
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1); os.dup2(devnull, 2)   # 释放管道写端引用
            p.stdin.close()                             # 最后一个写端 → EOF
            p.wait(timeout=10)
        except Exception:
            pass
    atexit.register(_drain)
    print(f"[logtee] 日志同步落盘 -> {path}\n[logtee] argv: {' '.join(sys.argv)}"
          f"\n[logtee] 代码位置: {os.path.abspath(__file__)}(排障用: 确认跑的是"
          f"挂载代码还是镜像内代码)")
    return path
