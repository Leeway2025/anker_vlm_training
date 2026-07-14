# 全流程 TPU 验证报告(v6e-4,2026-07-14/15)

目的:在真 TPU + 真权重上,把训练管线的**每个阶段和每条旁路**都以短跑
(2~4 个优化器步)方式打穿一遍,验证机制而非收敛质量。

环境:GCE ct6e(v6e-4)、gemma-4-e2b-it 真权重、64 条合成 euno-WDS
样本(16×384×384 JPEG,含伪造 Gemini 资产 A/C)、torch 2.9.0+cpu /
torch-xla 2.9.0 / transformers 5.13.0(requirements-lock 同款)。

## 结果总览

| # | 验证项 | 配置 | 结果 |
|---|---|---|---|
| R1 | stage b 主循环 + eval + checkpoint 保存 + 早停回调 + load_best | 单进程,4 步,eval/save 每 2 步 | ✅ exit 0;eval 跑 2 次,checkpoint-2/4 + final 齐全 |
| R2 | **多进程**(4 rank)+ eval + 存盘 rendezvous | 4 进程,2 步 | ✅ exit 0;单 rank 峰值 RSS 65GB,4 rank 合计 245GB(瞬时) |
| R3 | stage a(仅 projector,无 LoRA) | 单进程,2 步 | ✅ exit 0;adapter=False 分支正确,projector.pt 落盘 |
| R4a | run_inference **TPU 生成** | 4 样本,32 新 token | ✅(修复后)exit 0,4/4 产出;修复前 50 分钟 0 产出,见下 |
| R4b | hard_mining → sw.json → 加权续训 | 1 步 | ✅ 48→65 样本物理复制,exit 0 |
| R5 | aux_heads(资产 A) | 2 步 | ✅ exit 0;aux_heads.pt 落盘 |
| R6 | implicit_cot(资产 C) | 2 步 | ✅ exit 0;cot 数据路径 + 退火回调无错 |
| R7 | build_kto_data → kto.py(4 rank) | 1 epoch/76 batch | ✅(修复后);修复前 rank 间 batch 数不齐 → all_reduce 死锁,见下 |
| R8 | swa 权重平均 + pipeline 导航 | CPU | ✅ 平均 2 个 checkpoint,projector 附带;status/next/cmd 正常 |

主 OOM 修复(见 OOM_DIAGNOSIS.md)在 R1/R2/R4b/R5/R6 所有 Trainer
路径上生效:训练期单进程 RSS 稳定在 35~76GB 区间,无失控增长。

## 本轮修出的两个新缺陷

### 1. TPU 上 model.generate 逐 token 重编译(run_inference 不可用级)

现象:4 样本 50 分钟零输出,22 张越来越大的图在反复编译。
PT_XLA_DEBUG 定位:generate 循环每个解码步的停止判断
(stopping_criteria / `_has_unfinished_sequences`)强制读张量值,
而解码步之间没有图边界 → 惰性图逐 token 增长,每 token 编一张更大的图。

修复(training/inference_utils.py):注入 `_XlaStepMarker`
LogitsProcessor —— 它在每步前向之后、停止判断取值之前被调用,
在此处 `xm.mark_step()` 把图切到单解码步;配合已有 static cache,
图不再随 token 增长。实测 4 样本 20 分钟内全部产出(含全部编译)。

### 2. kto.py 多 rank all_reduce 死锁

现象:两次复现均在同一位置停住(最后日志 step10,log_every=10),
CPU 忙轮询 25 分钟以上无进展。
根因:`batches[rank::world]` 分片在 batch 总数不是 world 整数倍时
(78÷4 → 20/20/19/19),多出一步的 rank 在 `kl_all_reduce` /
梯度 all_reduce 里等不齐全员 → 集合通信死锁。
修复(training/kto.py):batch 总数向下对齐到 world 整数倍再分片,
各 rank 步数严格一致。

## 已知残留(不阻塞,建议排期)

- KTO 步内 4 次前向(policy/ref × 匹配/错配)在 bs1 下约 2.5 分钟/步,
  真实训练建议加大 per_device_batch_size 摊薄。
- TPU generate 修复后仍有 ~75 次小图编译(每样本长度差异导致 prefill
  变体);如需大规模推理,建议配 `xr.initialize_cache`(持久编译缓存)
  或按长度分桶。
- collator 缺 video_metadata(fps=24 假时间戳)、processor_kwargs
  告警刷屏 —— 见 OOM_DIAGNOSIS.md 第七节。
- 8 卡全局 batch 提醒:accum32 为 v6e-1 标定,8 rank 下全局 batch=256,
  建议 accum 降 4 或线性调 LR(OOM_DIAGNOSIS.md 第五节)。

## 复现方法

4 卡机 /dev/shm 下:`make_fake_wds.py` 生成合成数据;`run_repro.sh
<tag> <ckpt_on|off> <nprocs> <max_steps> [stage]` 驱动各阶段短跑
(train.py 的 REPRO_NPROCS / REPRO_MAX_STEPS 钩子);`run_infer.sh` /
`run_kto.sh` 驱动推理与 KTO。RSS 曲线由 rss_monitor.sh 每 2s 记录。
