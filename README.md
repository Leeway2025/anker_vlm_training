# Anker VLM 训练代码库(Gemma 4 E2B / TPU v6e / torch_xla + PEFT LoRA)

> 交付给客户的训练代码。客户用自己的 1M 数据按 `docs/REPRODUCE.md` 重训;
> 我方在代理数据集上完成代码验证(数据不出客户环境)。
> 方案依据: `../training_plan.md`(定稿版)。

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

## ⚠️ TPU 烟测清单(首次运行必查,本仓库在无 TPU/权重环境编写)

1. gemma-4 模型类名: `AutoModelForImageTextToText` 是否命中(common._load 有回退)
2. processor 的 video 入参签名(collator/inference_utils 标 SMOKE 处)
3. PLE 参数命名: freeze_base 打印 frozen_keyword_hits,`ple` 计数须 >0
4. `config.text_config.layer_types` 存在性(差异化 rank 的 global 层检测)
5. 固定 padding 生效: 训练 10 步观察 XLA 编译次数(应只编译 1~2 次)
6. PISSA 初始化(peft≥0.10),失败自动回退有 [WARN] 日志
7. kto.py 的 DataLoader 拼装(骨架就绪,依赖 2 的确认)

## 红线(代码级强制)

- base/PLE/Embedding 冻结断言(common.freeze_base 关键字未命中即 raise)
- 时序翻转禁止(augmentation.assert_monotonic,违者抛异常)
- 输出解析按位置取字段 + 大小写矫正 + RT×SubKS 合法组合校验

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
