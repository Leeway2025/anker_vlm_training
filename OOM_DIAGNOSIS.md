# v6e-8 主机内存 OOM:根因诊断与修复报告

日期:2026-07-14 · 结论状态:**根因已复现实锤,修复已验证**

## 一、现象

v6e-8(8 卡)上跑 Phase 5 stage b(bs1 × accum32),训练在第一个 step
完成之前整机卡死,dmesg 显示内核 OOM killer 介入:

- 被杀进程 anon-rss **229GB**(total-vm 334GB)
- 8 个训练 rank 同时失控,RSS 各为 229/219/195/195/194/148/99/57 GB,
  合计 ≈ **1.34TB**,打爆主机内存
- 无任何 `RESOURCE_EXHAUSTED`(HBM OOM)日志 → 不是显存问题
- 不是 spot 回收(有完整 dmesg 证据链)

## 二、根因

**transformers(≥4.46,现场 5.13)Trainer 的梯度累积实现与 torch_xla
惰性执行模型相冲,导致整个累积窗口被展开成单张巨型 HLO 图。**

因果链(每一环都有实测证据):

1. HF Trainer 自身不再调用 `xm.mark_step()`,注释明确写着
   "PyTorch/XLA relies on the dataloader to insert mark_step each
   iteration"(trainer.py L1802)——即依赖 accelerate 包装的
   `MpDeviceLoader` 在**每次取 batch 时**切图。
2. 但 `get_batch_samples()`(L1711)为了算 `num_items_in_batch`,
   会把整个累积窗口的 32 个 micro-batch **一次性从 dataloader 取完**。
   MpDeviceLoader 的 mark_step 全部在这个"纯取数期"提前触发。
3. 随后 32 次 forward + backward(+ 梯度检查点重算图)之间
   **再无任何图切割**,直到下一个窗口才遇到 mark_step。
4. XLA 收到一张 32 步展开的巨型图。编译器工作内存(host RAM,
   非 HBM)随展开步数爆炸,30 分钟编不出第一个 step。
5. 8 个 rank 各自编译各自的巨图、进度不同 → RSS 快照 57~229GB 不等,
   合计 1.34TB → 内核 OOM killer。

梯度检查点(XLA checkpoint patch)不是根因,只是放大器:
重算子图让巨图更大(225GB vs 136GB+),关掉它照样爆。

## 三、复现实验(v6e-4 单进程,假数据 64 样本,bs1,6 step 上限)

| 实验 | 配置 | 单进程 RSS 峰值 | 结果 |
|---|---|---|---|
| A | accum32 + ckpt **on** | **225.6GB** | 30 分钟 0 step,进程被杀 |
| B | accum32 + ckpt **off** | **>136GB**(仍在爬升时终止) | 20+ 分钟 0 step |
| C | accum**1** + ckpt off | **33.6GB,全程平稳** | 正常出 step(首步 171s 含编译) |
| D | accum32 + ckpt on + **本修复** | **76GB 瞬时峰值**(编译期,含 PT_XLA_DEBUG 开销),HBM 峰值 20.8/31.25GB | **3/3 step 完成,exit 0**,loss 正常,final/ 检查点完整落盘 |

- 实验 A 的 225.6GB 与客户被杀 rank 的 229GB 几乎一致 → 同一病灶。
- 实验 C/A 唯一变量是累积步数 → 图展开定罪。
- RSS 曲线形态:前 ~15 分钟稳定在 31~39GB(build/传输/编译前段),
  之后随编译推进指数式爬升直至打爆——这解释了为什么小规模冒烟
  测试"看起来正常":爆炸发生在启动后 15~30 分钟。

## 四、修复

`training/trainer.py`:`WeightedSFTTrainer.training_step()` 重载,
每个 micro-step 结束后手动 `xm.mark_step()`:

```python
def training_step(self, model, inputs, num_items_in_batch=None):
    if _XLA:
        xm.mark_step()   # 入口切图: 隔离上一窗口的 optimizer.step 子图
    loss = super().training_step(model, inputs, num_items_in_batch)
    if _XLA:
        xm.mark_step()   # 出口切图: 图上限 = 单个 micro-batch 的 fwd+bwd
    return loss
```

- 语义不变:梯度在 `.grad` 缓冲区跨图累积,这正是 torch_xla
  官方训练循环的标准写法。
- 出口切图解决主 OOM;入口切图另有必要:否则窗口尾部的
  optimizer.step/zero_grad 会融进下一窗口首个 micro-step 的图,
  该融合图随步数变化反复重编译(实测第 2 个 step 又编了 14 分钟)。
  两刀都切后 3 个 step 共 25 次编译、全部完成。
- 对 GPU/CPU 无影响(`_XLA=False` 时是空操作)。

## 五、给现场的操作指引

1. 拉最新 main 分支(含本修复)。
2. 配置无需改动:bs1 × accum32、gradient_checkpointing: true 均可保留。
3. 启动后前两个 step 慢(编译),此后应稳定;建议留一个
   `watch -n 5 'ps aux --sort=-rss | head -3; free -g'` 观察 10 分钟,
   单 rank RSS 应稳定在 ~35GB 量级(8 rank 合计 <300GB)。
4. 若未来升级 transformers,注意本修复依赖 `training_step` 签名
   `(model, inputs, num_items_in_batch)`;升级后跑一次 tests/。
5. **全局 batch 提醒**:`bs1 × accum32` 是在 v6e-1 单芯上标定的档位。
   8 卡上同一份 yaml 的全局 batch = 1 × 32 × 8 = **256**,是标定值的
   8 倍,LR 与收敛行为都会变。建议 8 卡把 `gradient_accumulation`
   降到 4(保持全局 32),或按线性法则同步调 LR。这与 OOM 无关,
   但会直接影响训练效果。

## 六、诊断中排除的其他假设(记录备查)

| 假设 | 排除依据 |
|---|---|
| HBM OOM | 无 RESOURCE_EXHAUSTED;死于内核 OOM killer |
| spot 抢占 | dmesg 有完整 OOM 证据链 |
| WDS shuffle buffer | euno_wds 无 shuffle buffer,JPEG bytes 存储,数据侧 <1GB |
| 动态 shape 重编译风暴 | 文本固定 padding、视觉 (16,630,768) 全静态 |
| Gemma4 内置全词表 CE | labels 已 pop,logits_window 生效(HBM 侧优化,与本次 host RAM 无关) |
| 梯度检查点 patch 是根因 | ckpt off 依旧爆(136GB+);它只放大不引爆 |
| dataloader worker 泄漏 | 客户 dmesg 进程表中无 worker 进程 |

## 七、顺带发现(未在本次修复,建议排期)

- collator 未传 `video_metadata`,transformers 回退 fps=24 生成帧时间戳,
  与真实采样(全片均匀 16 帧)不符 → prompt 里时间戳失真(质量问题,
  不影响稳定性;训练/推理两侧一致时影响有限,但建议统一修)。
- `processor.__call__` kwargs 告警刷屏(transformers 5.13 要求走
  `processor_kwargs`),纯日志噪音。
- `[fp32] 0 LoRA tensors upcast` 是预期行为(新版 peft 注入即 fp32),
  日志措辞易误读,可改为显式说明。
