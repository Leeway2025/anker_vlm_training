# 代码开发工作状态快照

> 更新:2026-07-09(第 15 轮,**真 v6e-8 八卡验证完成**,europe-west4-a spot,
> 用后即删)—— 此前"待客户环境确认"的 8 卡项全部关闭:
> - ✅ torch_xla.launch 8 进程,world=8,ordinal 连续
> - ✅ XLA 集合通信实测(CPU 模拟不可验层): all_reduce SUM 精确、
>   xm.optimizer_step 跨核梯度平均精确(w=-4.5)、rendezvous、
>   rank0 落盘唯一 —— REAL8 PASS
> - ✅ Trainer 8 核真训: **每核步数 = ceil(84/(8×2)) = 6 精确命中**
>   (DistributedSampler 切分正确),loss 8.74→6.18 平滑降,eval 正常,
>   [save] 仅 rank0 一次;跨阶段恢复(3a→3b projector)正常
> - ✅ **bs 显存定论: bs8/芯 OOM(34.8G/31.25G,超 3.55G);bs4 全流程
>   通过**(LoRA 512/256 全量,训练+eval+保存)→ base.yaml 默认改
>   bs4 + grad_accum 8(全局 256 与方案等效),REPRODUCE 同步
> - 经验入档: spawn 子进程重 import → 自定义启动脚本必须
>   `if __name__ == "__main__"` 守卫(train/kto 本有,临时脚本踩过);
>   spot 抢占 = 机器删除不可重启,产物须即时外存 + 断点续跑
> 剩余客户侧确认项仅剩: RKLLM 工具链(Issue #480、异构 rank)——
> 端侧职责,与 TPU 无关

> 更新:2026-07-09(第 14 轮,**训练部分整体 review: 5 硬伤修复 + 真机回归**)
>   ㉖ trainer 整段 .float() 物化 (B,L,262k) fp32 → bs8≈17GB v6e 必 OOM
>     → 分块加权 CE(ce_chunk=256,步数静态单编译)+ 监控量张量累积
>     (log 步才 .item(),消除每步 XLA 同步)【原版遗留,bs2 演练侥幸】
>   ㉗ 辅助头/KS 父类头**从未进优化器**(懒初始化晚于 build_optimizer,
>     随机权重陪跑)→ eager 初始化(fp32 头)+ aux/ks 分别空参守卫
>     + 单测(一步优化权重必变)。真机再踩两坑一并修:维度须在全部
>     匹配模块找 2D 权重(最深匹配可能是 norm);钩子须挂顶层 embedder
>     (最深层是投影前 768 维,与 1536 头错位)【原版遗留】
>   ㉘ pipeline.py 仍生成已消失的 xla_spawn 启动器【原版遗留】
>   ㉙ val_size 30000(1M 设定)→ 8000(100k 阶段)【原版配置】
>   ㉚ split_by_camera 把 camera_id=unknown 当同一机位整组切分 →
>     退化为按 video_id【euno 流程引入的交互问题】
> 真机回归(UCSD 沙盒 5b): 新 trainer 训练+eval 正常、loss_cls 记录,
>   final 四件套含真训 aux_heads.pt(1536 维 fp32);24/24 + 6/6 全绿

> 更新:2026-07-08(第 13 轮,**全阶段六维 review + "GT 质量>Gemini"原则贯穿**)
> 六维审计(数据来源/前置就绪/下降风险/输出/结束判定/文档)新修:
>   ㉒ euno_to_labels 不支持 gs://(裸 open)→ 复用 open_binary
>   ㉓ load_wds_frames 对 gs:// 明确报错+指引(训练随机读需本地盘/
>     gcsfuse;gs:// 直读仅标注流式作业支持)
>   ㉔ camera_fingerprint 支持 WDS 首帧(euno uuid 样本兜底路径打通)
>   ㉕ run_inference 加 --shard i/n(10 万级推理多进程/多机切片,
>     各片独立断点续跑)
> **原则贯穿(用户确认: 人工修正 GT 质量 > Gemini)**:
>   GT 全量信任;白名单只裁 Gemini 自产资产的采用范围,不裁样本:
>   - split_assets 过滤前移到资产层(白名单外条目不写入资产文件)
>   - phase5b/5d use_whitelist_only → false(全量样本;无资产样本
>     自动降级: aux=-100 / CoT 走生产模式,代码天然支持)
>   - KTO 数据构造客户数据不传 --whitelist
> 文档: WALKTHROUGH 新增"各训练阶段一览表"(数据来源/结束判定/产出/
>   下降防护与回退)+ 通用回退协议 + 监控集离线消费方式说明
>   (训练内早停=val eval_loss;监控集=阶段闸门,离线推理+metrics)

> 更新:2026-07-08(第 12 轮,**GT 来源事实确认: Gemini 标注+人工修正**)
> 连锁语义修正(代码参数不变,用法与解读反转):
> - hard mining **客户数据不要传 --whitelist**: GT 人工背书 → 错例
>   基本都真;Gemini 不认同 GT 的错例恰是人工看得出、Gemini 看不出的
>   盲区难例,过滤=错杀(代理集纯 Gemini 伪 GT 场景仍建议传)
> - gt 模式白名单存在**同源偏置**(二次 Gemini 与 GT 同源):语义收窄为
>   "资产 A/C 可信子集"(5b/5d 筛选用途不变),不可当 GT 对错的独立证据
> - discarded/gt_suspect 高频不一致对 = Gemini 系统性盲区(报我方改
>   标注 prompt),而非 GT 错标要客户重标;白名单率验收线上调至 ≥85%
> - 利好: h/n GT 噪声预期低于方案假设的 15%;KTO desirable 与
>   description 目标质量更可靠

> 更新:2026-07-08(第 11 轮,**hard mining 下滑风险加固**,用户提问触发)
> 朴素"错例×3"的三条整体下滑通路 → 三层防护落码:
>   ① 错例扎堆难类 → 类别先验被改写(自然分布测试集上净降)→
>     per-class 质量膨胀上限(--max-class-inflation 1.5 默认),
>     公式 w_c = min(w, 1+(cap-1)·n_c/k_c);"保持类别分布"从方案文字
>     变成代码强制
>   ② 错例混 GT 噪声(h/n ~15%)放大即教错 → --whitelist(资产 D)
>     只放大 Gemini 确认过 GT 的可信错例,wrong_untrusted 单独上报
>     (>30% 提示先数据审计)
>   ③ 难例主导 → 易样本遗忘 → 续训强制早停+load_best(已有),
>     验收 = 监控集总指标不降且热区混淆减少
>   配套: apply_hard_mining 分数权重改流式最大余数复制(直接 round
>   会把 <1.5 全截成 1,上限失效);per-class 膨胀报表输出

> 更新:2026-07-08(第 10 轮,**训练阶段衔接审计: 4 个文件级断点修复**)
> 逐衔接点重推全链(用户质疑触发),发现并修复:
>   ⑱ **资产拆分器缺失**: pass1.jsonl 的属性嵌在 gemini_output 里,
>     训练读顶层字段 → 辅助头/CoT 监督静默丢失。新增
>     annotation/split_assets.py(pass1+whitelist → asset_A/C/D),
>     已用真 API 标注产出验证与训练消费端对齐
>   ⑲ **推理不支持 WDS**: run_inference/generate_predictions 只读视频
>     文件,客户数据(WDS 帧)下 hard mining/KTO 前置推理跑不了 →
>     euno_wds.load_frames_for_record 存储自适应,推理/KTO 统一接入
>   ⑳ KTOVideoDataset 同病 → 同修(按 labels 的 meta.storage 分流)
>   ㉑ swa_final 缺 projector.pt → 终评静默用 base projector →
>     swa.py 自动从兄弟 final/ 附带(缺失时显式 WARN)
>   另: stage b 未传 --init-from 时 WARN(防静默从零训);
>   WALKTHROUGH 新增文件级衔接图(谁产出→谁消费),明确 7a 的 preds
>   必须用 5d final 重新推理(preds_v15),不能复用 v0 的 preds

> 更新:2026-07-08(第 9 轮,**接入客户真实数据规格 euno数据集说明.md**)
> 客户数据一手规格落地,修正多项纸面假设:
> - **真实 GT 分隔符无空格**("D|g|A man...")→ base.yaml separator="|",
>   5 条真实样例字节核对 5/5;prompt.txt 与客户模板**逐字节一致** ✓;
>   无空格上下文里 26 分类字母仍独立 token ✓;全代码默认值同步迁移
>   (含 build_kto_data —— 原本会按错误格式构造 desirable 串)
> - **客户数据 = WDS 分片(帧已上游均匀 16 帧 + 384×384)**:
>   新增 data/euno_wds.py(标注转换 + EunoWDSDataset,train.py 按
>   meta.storage 自动选择);增强受限——时序裁剪/原图 RandomCrop 失效
>   (拿不到原片),仅保留翻转/亮度/帧 dropout
> - **无 resolution 字段** → 4.2 节 view_type 分辨率免费打标规则不可用,
>   view_type 全靠 Gemini(方案影响,需知会客户)
> - **camera_id**: video_rel 内设备序列号(T8xxx)提取;uuid 命名 →
>   camera_fingerprint 兜底
> - **数据口径已定(2026-07-08 用户确认): 先用 balanced 100k 训练,
>   1M 自然分布训练后续第二轮再做**。连带影响:
>   ① 100k 阶段与 EunoVLM 同口径,对比公平;6.3"不做类别均衡"在此阶段
>     天然不适用(数据本身已均衡),1M 阶段回归自然分布原则
>   ② 训练/验证集(balanced)与冻结测试集(自然分布,m 占 27.5% vs
>     训练 12.7%)分布不一致是**有意设计** —— 早停用的 eval_loss 有
>     分布偏差,监控集指标解读须记住这一点
>   ③ 1M 阶段依赖客户提供 1M 标注文件(现有标注仅 balanced 100k;
>     wds_full 的 1,082,100 样本帧数据已在)
> - eval/euno_results_adapter.py: euno 推理结果 json(pred.result/score)
>   → metrics 输入,基线复现同口径
> - 全链 mini-WDS 回归测试(test_euno_wds_roundtrip),22/22 绿

> 更新:2026-07-08(第 8 轮,**三项遗留收口: KTO 实装 / generate / 分布式入口**)
> - **kto.py DataLoader 实装完成**(骨架清零): KTOVideoDataset(确定性
>   16 帧,无增强)+ KTOCollator(复用 AnkerCollator 的 prompt/视频编码,
>   completion 直接 tokenize)+ 主循环(分层 batch → kto_step →
>   xm.optimizer_step → 权重分化告警 → checkpoint/final CPU 安全保存,
>   projector 原样传递)。真机: step1 loss=0.624 / weight_gap=8.6e-06
>   (优化器首步即真实更新)
> - **分布式入口修正(bug #15)**: 文档所写 `torch_xla.distributed.xla_spawn`
>   在 torch_xla 2.x 不存在(老版 transformers 示例脚本)。train.py/kto.py
>   已内置 `torch_xla.launch(_mp_fn)`(v6e-8 自动 8 进程),README/REPRODUCE
>   启动命令同步更新;kto 数据按 global_ordinal shard,KL all_reduce 跨核
> - **generate 路径真机通过**: inference_utils 加 static KV cache
>   (XLA 动态 cache 逐位置重编译不可用)+ run_inference 上卡修复与
>   --max-new-tokens/--batch-size 参数;E2E: 加载→恢复→生成→解码→
>   写盘→metrics 全通(合成模型输出退化属预期,format_fail 兜底正确)
> - **两个手写 XLA 循环的致命坑(真机踩掉,已修)**:
>   ⑯ 循环缺 xm.mark_step() → 懒执行图跨步无界生长,步步重编译
>     (每步 ~10 分钟);加步界后稳态 ~30s/步,12 步 5 分钟跑完
>   ⑰ 日志用布尔掩码索引(动态形状)→ 改静态形状(掩码乘法)
> - KTO 真机 12 步: weight_gap 单调增长(8.6e-6 → 3.7e-5,优化器持续
>   真实更新);checkpoint-6/12 + final(含 projector 传递)落盘
> - 仍留客户侧首跑确认: 真 8 核(本机 v6e-1,world=1 已验证入口/shard/
>   all_reduce 代码路径)/ per-device batch 8 显存 / RKLLM 工具链

> 更新:2026-07-08(第 7 轮,**多 LoRA/NPU 部署红线加固**,S-LoRA 式
> 共享 base 场景,用户提出 + peft 官方警告佐证)
> - **PISSA 默认关闭**(base.yaml init_weights: gaussian): PISSA 初始化
>   会就地修改 base 权重(W←W−BA),与端侧多 adapter 共享同一 base.rkllm
>   直接冲突;且 peft 声明其"转回标准 LoRA"流程与 rsLoRA/rank_pattern 互斥
> - **build_lora 加 base 不变性哨兵**: 注入前采样 base 权重切片,注入后
>   逐字节比对,任何改 base 的初始化(pissa/olora/corda)当场 raise
> - **split_deliverables 加 rsLoRA 折叠**: √r 折进 lora_B(数值恒等),
>   导出 adapter 为纯标准 LoRA 语义 + use_rslora=false 配置
>   —— 不折叠的话 RKLLM 按 α/r 读会差 √r 倍(r=256 → 16 倍,静默精度崩)
> - 遗留 Phase 0 客户工具链测试项: RKLLM 是否接受逐层异构 rank(512/256);
>   不接受则 base.yaml 设 r_global=r_sliding 一行回退(统一 rank)
> - ✅ gaussian init 下复验通过: 5d(退火回调触发)→ SWA(2 ckpt 平均)
>   → 导出(rsLoRA 折叠 205 个 lora_B / llm_adapter 410 键 /
>   vision_merged 659 张量)→ metrics 全指标;base 哨兵静默通过

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
