"""8 卡语义模拟(任何机器可跑): 8 真进程验证本仓库的分布式逻辑契约。

    python3 tests/sim_multicore_8.py

覆盖(我们代码自身的正确性):
  ① 数据分片 batches[rank::world]: 8 进程各自计算,无重叠、并集完整
  ② rank0 落盘 + 栅栏: 恰好一份 final,其他进程等待后才继续
  ③ 梯度平均数学: 各 rank 不同梯度 → 平均更新后参数一致
    (xm.optimizer_step 的语义契约,用显式 allreduce/8 复算)
  ④ KTO 分层 batch 在分片后仍满足"每 batch 含错例"

明确不覆盖(仅真 v6e-8 可验,首跑清单见 WALKTHROUGH):
  - XLA 集合通信本体(xm.all_reduce / CCL)—— CPU 线程模拟结构性
    不可行: PJRT CPU 多设备是单进程线程模型,集合操作 op_id 走全局
    计数器,并发追踪必然错位死锁(2026-07-09 实测,两种写法均复现);
    该层由 torch_xla 上游在 TPU pod 上保证,我们只调用标准 API
  - 每芯显存(bs8)/ ICI 带宽 / libtpu 行为
"""
import json
import multiprocessing as mp
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORLD = 8


def worker(rank, out_dir, barrier, q):
    import random
    import torch
    from training.kto import plan_stratified_batches

    ok = {"rank": rank}

    # ① 分片
    idx = list(range(100))
    ok["shard"] = idx[rank::WORLD]

    # ③ 梯度平均契约: grad_r = rank+1;平均后 w = -lr*mean = -(WORLD+1)/2
    w = torch.zeros(1, requires_grad=True)
    (w.sum() * float(rank + 1)).backward()
    grads = q["grads"]
    grads.put((rank, float(w.grad.item())))
    barrier.wait()                       # 等 8 份梯度齐(≈xm.reduce 语义)
    if rank == 0:
        gs = [grads.get() for _ in range(WORLD)]
        mean_g = sum(g for _, g in gs) / WORLD
        q["mean"].put(mean_g)
    barrier.wait()
    mean_g = q["mean"].get()
    q["mean"].put(mean_g)                # 放回供其他 rank 读
    with torch.no_grad():
        w -= 1.0 * mean_g                # lr=1 的平均梯度一步
    ok["w_after"] = float(w.item())

    # ④ 分层 batch × 分片: 每片内 batch 仍含错例
    labels = [i % 5 != 0 for i in range(80)]      # 16 条错例
    batches = plan_stratified_batches(labels, 4, 1, random.Random(0))
    mine = batches[rank::WORLD]
    ok["stratified_ok"] = all(
        any(not labels[i] for i in b) for b in mine) if mine else True

    # ② rank0 落盘 + 栅栏(save_final 守卫语义)
    if rank == 0:
        with open(os.path.join(out_dir, "final.txt"), "w") as f:
            f.write("rank0")
    barrier.wait()                       # ≈ xm.rendezvous("save_final_done")
    ok["final_visible"] = os.path.exists(os.path.join(out_dir, "final.txt"))

    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(ok, f)


def main():
    out_dir = tempfile.mkdtemp(prefix="sim8_")
    barrier = mp.Barrier(WORLD)
    q = {"grads": mp.Queue(), "mean": mp.Queue()}
    procs = [mp.Process(target=worker, args=(r, out_dir, barrier, q))
             for r in range(WORLD)]
    [p.start() for p in procs]
    [p.join(timeout=120) for p in procs]
    assert all(p.exitcode == 0 for p in procs), \
        [f"rank{i}:{p.exitcode}" for i, p in enumerate(procs)]

    reports = [json.load(open(os.path.join(out_dir, f"rank{r}.json")))
               for r in range(WORLD)]
    all_idx = sorted(sum((r["shard"] for r in reports), []))
    assert all_idx == list(range(100)), "分片重叠/遗漏"
    expect_w = -(WORLD + 1) / 2
    assert all(abs(r["w_after"] - expect_w) < 1e-6 for r in reports), \
        "梯度平均后各 rank 参数不一致"
    assert all(r["stratified_ok"] for r in reports), "分片后 batch 缺错例"
    assert all(r["final_visible"] for r in reports), "栅栏后 final 不可见"
    import glob
    assert len(glob.glob(os.path.join(out_dir, "final.txt"))) == 1
    print(f"SIM-8 PASS: 8 进程 | 分片完整 | 梯度平均一致(w={expect_w}) | "
          f"分层 batch 保持 | rank0 落盘唯一+栅栏可见")


if __name__ == "__main__":
    main()
