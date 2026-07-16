"""批数据预取器: 线程池并行解码 + 双缓冲,把 host 时间藏进设备计算。

背景(1M 规模硬需求): JAX 8 芯 ~15 样本/s → host 每秒要解码 240 张
JPEG,同步单线程取数必然喂不饱芯片。PIL 解码与 numpy patchify 在 C 层
释放 GIL → 线程池即可真并行,免多进程的序列化开销(帧 7MB/样本)。

设计约束(写死在实现里,不是可选项):
  - 工作线程异常必须透传主线程 —— 否则表现为训练无声卡死
    (torch 侧 dataloader worker 的经典坑);
  - 样本顺序与同步版逐条一致(确定性/可复现);
  - flush(): CoT 退火切换时清掉队列中旧模式 batch,边界零滞后。
"""
import dataclasses
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _disable_ktyping():
    """关闭 kauldron @typechecked(gemma 预处理装饰器)。

    其 scope 栈是全局非线程安全的(scope.py `assert s == self`),
    多线程并发解码必炸;类型检查属开发期辅助,训练进程关闭无副作用,
    还省每次调用的检查开销。"""
    try:
        from kauldron.ktyping import config as ktc
        base = ktc.get_config(None)
        disabled = dataclasses.replace(base, typechecking_enabled=False)
        ktc.get_config = lambda source=None: disabled
        return True
    except Exception:  # noqa: BLE001 —— 库结构变化时退化为不并行报错
        return False


class BatchPrefetcher:
    """产出与 train_sft.collect() 完全同构的 host numpy 批。

    用法:
        pf = BatchPrefetcher(ds, order_fn, batch_size, workers=8, depth=2)
        batch = pf.next()      # (tokens, labels, weights, patches, pos, aux, ks)
        pf.flush()             # 数据语义变更(如退火)后清队列重取
        pf.close()
    order_fn(k) -> 全局第 k 个样本的 dataset 下标(预取器只管调度不管顺序)。
    """

    def __init__(self, dataset, order_fn, batch_size, workers=8, depth=2):
        if not _disable_ktyping():
            print("[prefetch] ⚠️ 未能关闭 ktyping,并发解码可能不安全")
        self.ds = dataset
        self.order = order_fn
        self.bs = batch_size
        self.depth = depth
        self.pool = ThreadPoolExecutor(max_workers=workers,
                                       thread_name_prefix="prefetch")
        self.q = queue.Queue(maxsize=depth)
        self.cursor = 0                     # 下一个要调度的全局样本序号
        self.epoch_flag = threading.Event()
        self.stop = False
        self.gen = 0                        # flush 代数: 旧代产物直接丢弃
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._assembler, daemon=True)
        self.thread.start()

    # ---- 内部 ----
    def _prepare_one(self, idx):
        from jax_impl.data import make_vision_input
        ex = self.ds[idx]
        pt, px, counts = make_vision_input([ex["frames"]])
        return (ex["tokens"], ex["labels"], ex["weights"],
                pt[0], px[0], ex["aux_labels"], ex["ks_label"], counts)

    def _assembler(self):
        while not self.stop:
            with self.lock:
                gen = self.gen
                start = self.cursor
                self.cursor += self.bs
            idxs = [self.order(start + j) for j in range(self.bs)]
            try:
                futs = [self.pool.submit(self._prepare_one, i) for i in idxs]
                parts = [f.result() for f in futs]      # 异常在此抛出
                batch = tuple(np.stack([p[k] for p in parts])
                              for k in range(7)) + (parts[0][7],)
                item = ("ok", gen, batch)
            except Exception as e:  # noqa: BLE001 —— 透传,绝不吞
                item = ("err", gen, e)
            while not self.stop:
                try:
                    self.q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    with self.lock:
                        if gen != self.gen:      # 期间被 flush → 丢弃本批
                            break

    # ---- 对外 ----
    def next(self):
        while True:
            kind, gen, payload = self.q.get()
            with self.lock:
                cur = self.gen
            if gen != cur:
                continue                        # flush 前的旧代产物,丢弃
            if kind == "err":
                self.close()
                raise RuntimeError("预取工作线程异常(已透传)") from payload
            return payload

    def flush(self, restart_at=None):
        """清队列并从 restart_at(缺省=当前 cursor)重取(退火切换用)。"""
        with self.lock:
            self.gen += 1
            if restart_at is not None:
                self.cursor = restart_at
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass

    def close(self):
        self.stop = True
        self.pool.shutdown(wait=False, cancel_futures=True)
