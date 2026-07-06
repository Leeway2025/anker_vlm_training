# 代码开发工作状态快照

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
