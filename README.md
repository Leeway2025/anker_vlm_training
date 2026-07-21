# Anker VLM 训练代码库(Gemma 4 E2B / TPU v6e / torch_xla + PEFT LoRA)

> 交付给客户的训练代码。客户用自己的 1M 数据按 `docs/REPRODUCE.md` 重训;
> 我方在代理数据集上完成代码验证(数据不出客户环境)。
> 方案依据: `../training_plan.md`(定稿版)。

> **JAX/TPU 路线(推荐,吞吐 ≈ torch 的 1.9×)已投产**: 入口
> [`jax_impl/README.md`](jax_impl/README.md),逐阶段用法
> [`jax_impl/USAGE.md`](jax_impl/USAGE.md)。与本 torch 路线数据/格式/
> 超参三层兼容,adapter 可双向互换;本 README 以下内容描述 torch 路线。

## 结构

```
configs/       超参与生产 prompt(prompt.txt 分隔符须与客户 GT 字节核对)
data/          taxonomy(类别单一来源)/ formatting(分类 token ×4 加权)
               sampling(均匀 16 帧 + resize 384 拉伸,照抄生产)/ augmentation / dataset
annotation/    gemini_labeler(3.1 Pro 标注)/ consistency_filter(白名单)
training/      train.py(Phase 5/5b/5d)/ trainer(加权 CE)/ common(冻结+LoRA)
               hard_mining / build_kto_data / kto / swa / inference_utils
eval/          metrics(P/R/ACC/混淆/热区)/ format_validator / monitor_set
export/        split_deliverables(LLM adapter + vision merge)/ export_onnx
docs/          REPRODUCE.md(客户手册)/ issue480_workaround.py
tests/         纯逻辑单测(python3 tests/test_core.py,无 torch 依赖)
jax_impl/      JAX/TPU 训练路线(独立实现,零 torch 依赖;见其 README)
```

## 训练流水(推荐: 用编排器,每阶段可选可跳)

```
① 编辑 configs/pipeline.yaml —— 每个阶段 enabled: true/false
   跳过的阶段链条自动缝合(下一阶段接最近的已启用阶段产出)
   依赖自动校验(aux 需资产 A,cot 需资产 C,缺失时明确报错)
② python -m training.pipeline --dry-run   # 先看执行计划
③ python -m training.pipeline             # 顺序执行
   中途失败: 修复后把已完成阶段设 enabled: false 续跑
   已有外部 checkpoint: 设 start_from_checkpoint 从任意点起步
```

## 手动逐阶段执行(等价,与 training_plan Phase 对应)

```
1. annotation:  gemini_labeler(客户 1M: --mode gt 过滤;代理集: double)
2. Phase 5:     train.py --phase configs/phase5_sft.yaml --stage a(warmup)
                → --stage b;中途 inference_utils + hard_mining.py 产权重续训
3. Phase 5b:    train.py --phase configs/phase5b_aux.yaml(7 辅助头)
4. Phase 5d:    train.py --phase configs/phase5d_cot.yaml(隐式 CoT+退火)
                → eval 验证 think 泄漏率 = 0%
5. Phase 6:     inference_utils → build_kto_data.py → kto.py
6. SWA:         swa.py --ckpts outputs/phase6_kto/checkpoint-*
7. 导出:        split_deliverables.py [--issue480] → export_onnx.py
8. 评测:        eval/metrics.py(--pred/--gt),对齐客户口径
```

## ✅ TPU 烟测清单(2026-07-07 已在 v6e-1 真机全部验证,9/9 PASS)

> `PJRT_DEVICE=TPU python tests/smoke_tpu.py` 可随时回归。真机确认结果:

1. ✅ 模型类名 `Gemma4ForConditionalGeneration`(AutoModelForImageTextToText 命中)
2. ✅ processor 签名: chat template `{"type":"video"}` + **`do_sample_frames=False`**
   (内置采样器默认重采 32 帧!)→ `pixel_values_videos`/`video_position_ids`/
   `mm_token_type_ids`;70 token/帧 ×16=1120;帧时间戳写进 prompt
   (无 video_metadata 时 fps=24 兜底,**须与客户生产端约定对齐**)
3. ✅ PLE 实名 `per_layer_*`(108 个);Projector 实名 `embed_vision.embedding_projection`
4. ✅ `layer_types` 存在: 35 层,global = [4,9,14,19,24,29,34](7 层,方案假设成立)
5. ✅ 固定 padding + 视频前向反向端到端(加权 CE loss 正常,无重编译)
6. ✅ PISSA 成功(target 正则指向 vision 的 `.linear` 后);rank_pattern
   512/256 生效;LoRA llm=410 / vision=192 张量,audio 塔已排除
7. ✅ kto.py 全链路(双 adapter / 4 forward / 反向 / 优化器更新)真机跑通

**真机烟测修掉的坑**(详见 WORK_STATUS 第 4 轮):processor 二次采样 /
ClippableLinear 注入 / XLA checkpoint 三连坑(补丁位置、model.train()、
vision 梯度静默丢失)/ get_peft_model 重冻 projector / KTO logprob 窗口 /
bf16 图间噪声下的监控信号设计。

## 红线(代码级强制)

- base/PLE/Embedding 冻结断言(common.freeze_base 关键字未命中即 raise)
- 时序翻转禁止(augmentation.assert_monotonic,违者抛异常)
- 输出解析按位置取字段 + 大小写矫正 + RT×SubKS 合法组合校验
  (非法组合表含 D|q,涉安全类升级为 C 告警,training_plan 14.2)

## 与 training_plan 的已知偏离(交付评审需知)

1. **KTO 实现**: 方案 10.2 写 TRL KTOTrainer,实际为自实现
   (training/kto.py)。2026-07 调研结论(支撑该决策):
   - TRL KTOTrainer 无 TPU/torch_xla 支持(CI 仅 GPU/CPU,XLA 路径失修),
     且不支持视频输入(v1.6.0 起仅图像);KTO 刚经历 experimental
     降级-重构-回稳定的动荡期,接口不稳
   - EasyDeL 是唯一 TPU 原生 KTO(JAX),但纯文本管线且偏离路线 A
   - 本实现的双 adapter ref 设计与 TRL v1.x 官方机制一致
     (TRL 对 PeftModel 自动复制 default → 冻结 "ref" adapter)
   - 损失数学与 TRL 逐项对齐(错配对 KL + 跨核 all_reduce + clamp),
     tests/test_kto_math.py 与 TRL 公式数值对拍,上 TPU 前必须全绿
   - **Plan B(如客户可用 GPU)**: TRL v1.6+ 的 KTO 多图支持理论上可把
     16 帧当图像序列喂入(Gemma 视频本就是图像序列),单 GPU 训 1 epoch
     白名单子集可行;受 TRL 重构期接口稳定性制约,仅作兜底

## Phase ↔ 代码映射(training_plan.md 对照)

| Phase | 配置 | 核心代码 | 产出 |
|---|---|---|---|
| 0 验证准备 | — | eval/metrics.py(基线+混淆矩阵)/ docs/issue480_workaround.py / tests/ | 基线数字 |
| 1 数据准备 | base.yaml data.* | build_dataset.split_by_camera / eval/monitor_set.py | 切分+监控集 |
| 2 Gemini 标注 | — | annotation/gemini_labeler.py → consistency_filter.py | 资产 A/C/D |
| 3 采样 | base.yaml sampling | data/sampling.py + augmentation.py | 帧张量 |
| 4 LoRA | base.yaml lora/lr/freeze | common.freeze_base/build_lora/build_optimizer | 可训模型 |
| 5 基础 SFT | phase5_sft.yaml | train.py --stage a/b + trainer.py + hard_mining.py | v0 |
| 5b 辅助头 | phase5b_aux.yaml | train.py + common.AuxHeads(taxonomy 词表) | v1 |
| 5d 隐式 CoT | phase5d_cot.yaml | train.py + formatting.build_cot_target + AnnealCallback | v1.5 |
| 6 KTO | phase6_kto.yaml | inference_utils → build_kto_data.py → kto.py(双 adapter) | v2 |
| 9 SWA | — | training/swa.py | v2_final |
| 11 导出 | — | export/split_deliverables.py → export_onnx.py | llm_adapter/ + onnx |
| 监控 | — | eval/metrics.py + trainer loss_cls/loss_desc 曲线 | 指标报告 |
| 端侧兜底 | — | format_validator.deployment_guard(参考实现) | accept/flag/escalate |
