# 客户视角分步演练记录(WALKTHROUGH)

> 在 TPU v6e-1 + 代理数据集(UCSD ped1/ped2,98 clips,双机位)上,
> 按客户执行顺序把 REPRODUCE.md 每一步真实走通。每步给出:
> **目的 / 输入 / 输出 / 注意事项 / 实测结果**。
> 演练用小步参数(bs2、0.2~1 epoch);客户正式训练用 REPRODUCE.md 默认值。
> 标签为规则伪标(`meta.label_source="rule_pseudo"`),资产 A/C/D 为模拟
> (记录含 `simulated:true`)—— 正式代理训练前须 gemini_labeler 重打。

## Step 0 环境自检

- **目的**: 确认依赖齐全、纯逻辑测试全绿,才允许碰 TPU。
- **输入**: 无(代码库本身)。
- **输出**: 测试通过报告。
- **注意事项**: ① torch 无"TPU 版"——正确栈 = `torch(+cpu 构建)` +
  `torch_xla[tpu]`,两者版本号必须一致(ABI 锁定);② 个别用例需 torch;
  ③ tensorboard 是必装依赖(Trainer 默认 report_to)。
- **实测**: `python3 tests/test_core.py` → **20/20 passed**;
  `tests/test_kto_math.py` → 6/6(KTO 数学与 TRL 公式对拍)。

## Step 1 数据准备

### 1a 数据落盘(客户格式)

- **目的**: 视频 + labels.jsonl 按约定字段落盘,元数据完整。
- **输入**: 原始视频(演练: UCSD tif 帧序列)。
- **输出**: `videos/*.mp4` + `DATA/labels.jsonl`
  (video_id / video_uri / duration_sec / resolution / labels / meta.camera_id)。
- **注意事项**: ① 验证集按机位/住户切分防"记住这个门廊"泄漏;
  **若客户数据没有机位/住户字段**,用 `data/camera_fingerprint.py`
  从首帧背景指纹自动聚类伪 camera_id(固定机位背景即天然指纹),
  否则 split 退化为按 clip 随机切、验证指标虚高;
  ② 时长应落在生产区间 5~40s;③ 分辨率原样记录(view_type 规则用)。
- **实测**: `data/prepare_proxy_ucsd.py` → 98 clips(ped1 70 + ped2 28),
  360×240 / 238×158,12~20s;`camera_fingerprint` 自动聚类 → **2 个伪机位
  (70/28),与真实机位零混淆**。

### 1b 格式字节核对(开训前红线)

- **目的**: 训练目标串与客户 GT **逐字节一致**(空格/分隔符/换行),
  且分类字母在 tokenizer 中落点稳定。
- **输入**: 从数据文件**原样复制**的 3+ 条 GT 串(不要手敲)。
- **输出**: 全部 OK 才开训;FAIL 按提示改 `format.separator`。
- **注意事项**: PDF 里的 prompt 代码块含排版产物,configs/prompt.txt 是
  清理版——须整串核对而非只看分隔符。
- **实测**: `eval/check_format_alignment.py --tokenizer google/gemma-4-e2b-it`
  → 3/3 字节一致;**26 个分类字母全部落在独立 token 上**(无合并风险)。

### 1c 分层监控集

- **目的**: 安全关键类/边界对样本在监控集中有统计意义(随机 30k 里
  可能只有几十条)。
- **输入**: labels.jsonl。 **输出**: monitor.jsonl。
- **注意事项**: 演练数据只有 m/s 两类,配额自动降级为自然分布补齐;
  客户 1M 数据上 q/r/u/n/j 各 300、边界对各 500 才会真实生效。
- **实测**: `python -m eval.monitor_set` → total=98。

## Step 2 Gemini 增强标注(资产 A/C/D)

- **目的**: 属性标签(A,喂辅助头)/ 推理链(C,喂隐式 CoT)/
  GT 一致白名单(D,5b/5d/KTO 数据筛选)。
- **输入**: labels.jsonl + 视频;Gemini API(客户环境 3.1-pro)。
- **输出**: asset_A/C.jsonl + asset_D_whitelist.txt。
- **注意事项**: ① `max_output_tokens` 须 8192(thinking 模型的思考 token
  吃输出预算,900 会截断 JSON——真 API 集成测试踩过);② 白名单率应 ≥80%,
  discarded 抽查可能是 GT 错标;③ 断点续跑安全。
- **实测**: 本环境无 API key —— **用规则模拟资产代跑**(记录含
  `simulated:true`,README_SIMULATED.txt 说明),whitelist=98(全量)。
  labeler 本体此前已用真 API(cloud-llm-preview1 / 2.5-pro)集成验证。

> **Step 3~8 说明**: 全部机制已在第 5/6/8 轮真机验证通过
> (烟测 9/9 + 合成数据 E2E 全链 + KTO 12 步 + generate,见 WORK_STATUS)。
> 本演练在 UCSD 代理数据上重跑同一链条以记录客户视角结果,
> 后台执行中(/tmp/anker_walkthrough/logs_chain.log),完成后回填实测行;
> 未回填处以 WORK_STATUS 第 6/8 轮结果为准。

## Step 3 Phase 5 基础 SFT

- **目的**: stage a 只训 Vision Projector(warmup);stage b 全 LoRA +
  projector 联合,得 v0。
- **输入**: labels.jsonl + 视频;stage b 需 stage a 的 final/。
- **输出**: `outputs/phase5_sft_a/final`、`outputs/phase5_sft_b/final`
  (adapter_model.safetensors + adapter_config.json + projector.pt)。
- **注意事项**: ① 首步 XLA 编译 5~8 分/图属预期,10 步后仍慢查 padding;
  ② 观察 loss_cls 曲线必须在降(不能只有 loss_desc 降);③ PISSA 禁用
  (多 LoRA 红线,build_lora 有 base 不变性断言);④ 启动命令就是
  `PJRT_DEVICE=TPU python training/train.py ...`(torch_xla.launch 内置)。
- **实测**: (演练进行中,待补)

## Step 4 推理 + Hard Mining

- **目的**: 用 v0 对训练集推理,错例 ×3 物理复制续训(保持类别分布)。
- **输入**: phase5_sft_b/final;labels.jsonl。
- **输出**: preds_v0.jsonl → sw.json → `outputs/phase5_sft_hm/final`。
- **注意事项**: ① XLA 推理必须 static KV cache(动态 cache 逐 token
  重编译不可用,inference_utils 已内置);② 错例判定只看分类字段;
  ③ preds 断点续跑安全。
- **实测**: (演练进行中,待补)

## Step 5 Phase 5b 辅助头(消费资产 A)

- **目的**: 7 属性头挂 Projector 输出,把身份/动作/时长知识蒸进视觉表征。
- **输入**: hm(或 5_b)final + asset_A + 白名单。
- **输出**: `outputs/phase5b_aux/final`(含 aux_heads.pt,不进交付物)。
- **注意事项**: 低置信度(<0.5)属性样本 aux loss 权重 0;
  只用白名单样本。
- **实测**: (演练进行中,待补)

## Step 6 Phase 5d 隐式 CoT(消费资产 C)

- **目的**: 推理链蒸馏(60% [REASON] 模式)+ 末段纯生产模式退火,
  防 `<think>` 泄漏。
- **输入**: 5b final + asset_C + 白名单。
- **输出**: `outputs/phase5d_cot/final`(v1.5)。
- **注意事项**: 验收必测 think 泄漏率 = 0%(eval/metrics think_leak_rate)。
- **实测**: (演练进行中,待补)

## Step 7 KTO + SWA

- **目的**: 纯分类偏好精修(desirable=GT 串 / undesirable=分类错误输出),
  SWA 平均最后 N 个 checkpoint。
- **输入**: v1.5 final + preds + 白名单 → kto_data.jsonl。
- **输出**: `outputs/phase6_kto/checkpoint-*`、`outputs/swa_final`。
- **注意事项**: ① 监控 weight_gap 必须增长(优化器真实更新哨兵);
  ② 每 100 step 监控集分类指标降 >0.5 → 停(classification_brake);
  ③ ref_gap/logratio 有 bf16 图间噪声,不作零检测用。
- **实测**: (演练进行中,待补)

## Step 8 导出交付物 + 评测

- **目的**: 拆 LLM adapter(交 rkllm-toolkit)与 vision merge
  (交 rknn-toolkit2);评测报告。
- **输入**: swa_final + base 权重 + 训练后 projector.pt。
- **输出**: `deliverables/llm_adapter/`(rsLoRA 已折叠为标准 LoRA 语义,
  use_rslora=false)+ `vision_merged.pt`;metrics_report.json。
- **注意事项**: ① --projector 必传,否则 Phase 5 projector 成果丢失
  (脚本有 WARN);② issue480 重命名按需;③ 辅助头天然不进交付物。
- **实测**: (演练进行中,待补)

## 已知与正式训练的差异(演练局限)

| 项 | 演练 | 客户正式 |
|---|---|---|
| 数据 | 98 clips 规则伪标 | 1M 真 GT |
| 资产 A/C/D | 规则模拟 | Gemini 3.1-pro 双温度过滤 |
| 步数 | 0.2~1 epoch | REPRODUCE 默认 + 早停 |
| 硬件 | v6e-1 单芯 | v6e-8(8 核,bs8 显存待首跑确认)|
| 指标意义 | 仅机制验证 | 真实质量指标 |
