# 代码开发工作状态快照

> 更新:2026-07-07(第 6 轮,**多阶段流水线 E2E 在 TPU v6e-1 真机跑通**)
> 合成 12 条视频小数据集,完整执行: Phase 5a(projector warmup)→
> 5b(LoRA 联合)→ 5b_aux(7 辅助头)→ 5d(隐式 CoT,退火回调真机触发)
> → hard_mining → build_kto_data → SWA(平均)→ split_deliverables
> (llm_adapter 410 键/剥离 192 vision 键 + issue480 重命名 +
> vision_merged.pt 659 张量含训练后 projector)→ eval/metrics 全指标。
> 跨 phase 恢复链(adapter unexpected=0 + projector)逐段验证通过。
> E2E 新修 7 个纯逻辑测试抓不到的 bug:
>   #8  优化器在模型上卡前创建 → Trainer 设备校验拒绝(train.py 已修)
>   #9  输出目录命名 outputs/5_sft_b 与 init_from/pipeline 约定
>       outputs/phase5_sft_b 断链(train.py 已修)
>   #10 save_final 对 XLA 张量直接 safetensors 序列化崩溃 → 显式搬 CPU
>   #11 projector 升 fp32 后与 bf16 embedding 在视频 masked_scatter
>       相遇,XLA 报 mixed precision → fp32 master 仅用于 LoRA;
>       projector 保持 bf16(lr 5e-4 远超 bf16 舍入分辨率,更新不丢)
>   #12 AnnealCallback 未继承 TrainerCallback → on_train_begin 崩(已修)
>   #13 swa.py copytree 连 optimizer.pt(3~4G/ckpt)一起拷 → 真机 ENOSPC
>       → 只拷 adapter 两件套(已修+小样回归)
>   #14 split_deliverables 的 issue480 导入用 cwd 相对路径 "docs" →
>       换目录运行即炸 → 改仓库相对路径(已修)
> E2E 范围外(仍未真机验证): generate 推理路径(XLA 逐位置编译,
>   用伪造 preds 替代验证了下游)/ kto.py DataLoader 实装 / 8 核分布式 /
>   export_onnx.py / hard-mining --sample-weights 真机续训
> 运维事实: v6e-1 首步编译 6~8 分钟/图、每 phase 15~25 分钟;
>   PJRT 持久编译缓存反序列化不支持(每进程必重编)—— 客户排期须计入

> 更新:2026-07-07(第 5 轮,**TPU v6e-1 真机烟测完成,9/9 PASS**)
> `tests/smoke_tpu.py` 全绿 + KTO 单步全链路(双 adapter/4 forward/
> 反向/优化器更新)真机跑通。真机发现并修复:
>   ① processor 内置采样器默认重采 32 帧 → collator/inference_utils
>     统一 `do_sample_frames=False`;输出键为 pixel_values_videos/
>     video_position_ids/mm_token_type_ids(collator 已重写)
>   ② vision tower 为 Gemma4ClippableLinear 包装 → LoRA target 改正则
>     指向内部 `.linear`;audio_tower 排除;PISSA 由此成功不再回退
>   ③ XLA gradient checkpointing 三连坑: 补丁须打在
>     transformers.modeling_utils(早绑定符号)/ 必须 model.train()
>     (eval 下 checkpoint 分支不进图)/ reentrant 段输入无梯度时
>     **vision LoRA 梯度静默全 None**(加 require-grad 钩子修复)
>     → 修复前 B2/L2048 视频反向 35.4G OOM,修复后 loss=9.099 正常
>   ④ get_peft_model 会重冻 freeze_base 打开的 projector
>     (Phase 5/5b 将静默不训 projector)→ build_lora 内恢复
>   ⑤ KTO sum_logprob 全长计算真实形状 OOM → 静态 completion 窗口
>     (prompt 锁死 → 位置静态)+ 窗口越界 clamp
>   ⑥ bf16 XLA 图间噪声 ~0.2 → "logratio 应为 0"类输出空间检测不可行,
>     ref_divergence_alert 改用权重空间(起点精确 0,一步后 9.1e-6>0 已验证)
> 客户对齐新增项: Gemma4 把帧时间戳写进 prompt,无 video_metadata 时
>   fps=24 兜底 —— 生产端时间戳/metadata 约定须确认,训练照抄
> 剩余未真机验证: train.py 完整 Trainer 循环 / 8 核 xla_spawn 分布式 /
>   generate 推理路径 / KTO DataLoader 实装(组件均已单测)/
>   per-device batch 8 的显存(仅验证过 B2;v6e 单芯 32G,建议先
>   bs2~4 + 加大梯度累积起步)

> 更新:2026-07-07(第 4 轮,KTO 加固 + 交付前修补)
> 本轮变更:
>   ① kto.py 风险修复:损失数学对齐 TRL(错配对 KL + all_reduce + clamp,
>     替换简化版 batch-mean z0)/ 分块 logprob(防全词表 fp32 物化 ≈17GB
>     OOM)/ 分层 batch(每 batch 固定混入错例)/ ref 失效告警 +
>     classification_brake 刹车;tests/test_kto_math.py CPU 对拍(上 TPU
>     前必须全绿)。调研结论:KTO×视频 VLM×TPU 无现成实现
>     (TRL 无 TPU 且无视频;EasyDeL 有 TPU KTO 但 JAX 纯文本),
>     详见 README 偏离说明(含 GPU+TRL Plan B)
>   ② eval/check_format_alignment.py:GT 整串字节核对 + 字母 tokenizer
>     检查(prompt.txt 是 PDF 排版产物的清理版,须整串核对非只分隔符)
>   ③ D|q 进非法组合表(training_plan 14.2,升级为 C 告警)
> ⚠️ trainer.py 的 SFT loss 仍有同款全词表 fp32 物化(s_logits.float()),
>   烟测时优先确认显存;必要时复用 kto.sum_logprob 的分块思路

> 更新:2026-07-06(第 3 轮,代码库完成)
> ✅ 框架决策已定:**torch_xla + HF/PEFT(路线 A)**,LoRA 不用全参,
>   MaxText text-only 路线放弃(用户确认)。
> ✅ 全部模块已写完;纯逻辑 18 项测试/冒烟全部通过;全仓 py_compile 通过。
> ⏭️ 下一步:TPU 真机烟测(README 的 7 项清单)→ 代理数据集采集启动。

## 完成度

| 模块 | 文件数 | 验证 |
|---|---|---|
| configs | 6 | 人工核对 |
| data(taxonomy/formatting/sampling/augmentation/build_dataset) | 5 | 纯逻辑全测 |
| annotation(gemini_labeler/consistency_filter) | 2 | validate_record 测过 |
| training(common/trainer/train/hard_mining/build_kto_data/kto/swa/inference_utils) | 8 | 纯逻辑测过 + 编译通过 |
| eval(metrics/format_validator/monitor_set) | 3 | 全测 |
| export(split_deliverables/export_onnx) | 2 | split 逻辑测过 |
| docs(REPRODUCE/issue480) | 2 | rename 逻辑测过 |
| tests | 13 用例 + 8 冒烟 | **全绿** |

## TPU 真机烟测清单(README 同步,首跑必查)

1. gemma-4 模型类名(AutoModelForImageTextToText 回退链)
2. processor video 入参签名(collator/inference_utils SMOKE 标记处)
3. PLE 参数命名(freeze_base 的 frozen_keyword_hits 必须 ple>0)
4. config.text_config.layer_types(global 层检测)
5. 固定 padding 生效(XLA 编译次数 1~2 次)
6. PISSA 回退日志
7. kto.py DataLoader 拼装(骨架就绪,依赖 2)

## 真 API 集成测试(2026-07-06,cloud-llm-preview1 / gemini-2.5-pro)

```
✅ label_one 全链路: 视频上传 → JSON 响应 → validate_record 零错误
✅ 发现并修复真 bug: max_output_tokens 900 → 8192
   (thinking 系模型的思考 token 消耗输出预算,900 导致 JSON 截断,
    mock 测不出,3.1-pro 同为 thinking 模型必踩)
✅ CLI 双温度两 pass(0.1/0.4)→ double 一致性过滤 → 白名单+伪 GT 正确
✅ gt 模式分流正确(Gemini C|s vs 人工 GT D|l → 正确进 discarded)
✅ 断点续跑: 重跑 0 次 API 调用,3.2 秒返回
✅ main() 增加 --vertex-project/--location(Vertex 环境支持)
📊 单条延迟 ~14s(2.5-pro thinking)→ 50 并发 1M 条约 3~4 天,符合预估
⚠️ cloud-llm-preview1 无 gemini-3.1-pro,集成测试用 2.5-pro;
   客户环境切 3.1-pro 只改 --model 参数
```

## 已知限制

- kto.py 主循环是骨架(DataLoader 拼装等 processor 签名确认后 1 小时工作量)
- inference_utils 的 processor 调用姿势同样待烟测
- prompt.txt 分隔符待客户 3 条 GT 字节核对
