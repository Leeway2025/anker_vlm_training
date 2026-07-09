# 客户视角分步演练记录(WALKTHROUGH)

> 在 TPU v6e-1 + 代理数据集(UCSD ped1/ped2,98 clips,双机位)上,
> 按客户执行顺序把 REPRODUCE.md 每一步真实走通。每步给出:
> **目的 / 输入 / 输出 / 注意事项 / 实测结果**。
> 演练用小步参数(bs2、0.2~1 epoch);客户正式训练用 REPRODUCE.md 默认值。
> 标签为规则伪标(`meta.label_source="rule_pseudo"`),资产 A/C/D 为模拟
> (记录含 `simulated:true`)—— 正式代理训练前须 gemini_labeler 重打。

## 数据流总览(每步的命令、输入 → 输出、如何接入自己的数据)

**客户真实数据(euno WDS 格式)接入——一条命令转换后全链通用**:
> 数据口径(已确认): **第一轮用 balanced 100k 训练**(与 EunoVLM 同口径);
> 1M 自然分布训练为后续第二轮,届时只换 `--annotation` 指向 1M 标注文件
> (帧数据 wds_full 已含 1M 样本;1M 标注需客户提供)。
```
python -m data.euno_wds --annotation euno_train_v3.0.18_balanced_100k_frames.json \
    --wds-dir anker_video_clips_wds_full --out DATA/labels.jsonl
# camera_id 自动从设备序列号提取;uuid 命名的补跑 camera_fingerprint
# 基线复现: python -m eval.euno_results_adapter --results <euno推理结果.json> \
#     --out-dir baseline_eval/ && python -m eval.metrics --pred ... --gt ...
```
(帧已上游预处理 → 训练自动走 EunoWDSDataset,时序裁剪/原图 crop 增强
自动禁用;分隔符已按真实 GT 核对为 `|` 无空格。)

**接入自己的数据只需两件事**:
1. 视频文件放一个目录;labels.jsonl 按 Step 1a 的字段写好(每行一条);
2. 改 `configs/base.yaml` 的 `data.*` 六个路径(labels_file / video_root /
   attributes_file / reasoning_file / whitelist_file / val_holdout_key)。
   之后所有命令**不带数据参数**的,都从 base.yaml 读;带 `--labels/--out`
   参数的,直接在命令行指过去。

| 步 | 命令(可直接复制) | 输入 | 输出 |
|---|---|---|---|
| 0 | `python3 tests/test_core.py` | 代码库 | 全部通过(当前 23/23)|
| 1b | `python -m eval.check_format_alignment --gt-samples gt3.txt --tokenizer <模型路径>` | 3+ 条原样复制的 GT 串 | 通过/FAIL 提示 |
| 1b' | `python -m data.camera_fingerprint --labels <labels.jsonl> --video-root <视频目录>` | 无机位字段的 labels | meta.camera_id 回写 |
| 1c | `python -m eval.monitor_set --labels <labels.jsonl> --out monitor.jsonl` | labels | 分层监控集 |
| 2 | `python -m annotation.gemini_labeler --labels <labels> --video-root <目录> --out pass1.jsonl --model <m> [--vertex-project P --location L] --temperature 0.1`(0.4 再跑一遍) | labels+视频 | gemini_pass{1,2}.jsonl |
| 2' | `python -m annotation.consistency_filter --mode double --gemini pass1.jsonl --gemini2 pass2.jsonl --out-dir filtered/`(有人工 GT 用 `--mode gt --gt labels.jsonl`) | 两遍标注 | filtered/whitelist_ids.txt 等 |
| 2'' | `python -m annotation.split_assets --gemini pass1.jsonl --whitelist filtered/whitelist_ids.txt --out-dir DATA/` | 标注+白名单 | **DATA/asset_{A,C,D}**(训练直接消费) |
| 3a | `PJRT_DEVICE=TPU python training/train.py --phase configs/phase5_sft.yaml --stage a` | base.yaml data.* | outputs/phase5_sft_a/final |
| 3b | `… --stage b --init-from outputs/phase5_sft_a/final` | 3a final | outputs/phase5_sft_b/final |
| 4a | `PJRT_DEVICE=TPU python -m training.run_inference --ckpt outputs/phase5_sft_b/final --labels <labels> --out preds_v0.jsonl` | 3b final | preds_v0.jsonl |
| 4b | `python -m training.hard_mining --preds preds_v0.jsonl --labels <labels> --out sw.json` → `train.py --stage b --init-from outputs/phase5_sft_b/final --sample-weights sw.json --output outputs/phase5_sft_hm` | preds+labels | sw.json → hm/final |
| 5 | `… train.py --phase configs/phase5b_aux.yaml [--init-from outputs/phase5_sft_hm/final]` | labels 全量 + 资产 A(资产层已过滤) | outputs/phase5b_aux/final |
| 6 | `… train.py --phase configs/phase5d_cot.yaml` | labels 全量 + 资产 C(资产层已过滤) | outputs/phase5d_cot/final |
| 7a | `python -m training.build_kto_data --preds <v1.5 preds> --labels <labels> --out outputs/kto_data.jsonl`(客户数据不传 --whitelist;代理集伪 GT 才传) | preds+labels | kto_data.jsonl |
| 7b | `PJRT_DEVICE=TPU python training/kto.py --config configs/phase6_kto.yaml` | kto_data + 5d final | outputs/phase6_kto/{checkpoint-*,final} |
| 7c | `python -m training.swa --ckpts 'outputs/phase6_kto/checkpoint-*' --out outputs/swa_final` | KTO ckpts | swa_final |
| 8 | `python -m export.split_deliverables --adapter outputs/swa_final --base <base权重> --projector outputs/phase5d_cot/final/projector.pt --out deliverables/ [--issue480]` | swa_final | llm_adapter/ + vision_merged.pt |

### 训练阶段衔接图(文件级,谁产出 → 谁消费)

```
DATA/labels.jsonl(1a/2A①)──────────────┬─→ 3a/3b/5/6/7b 训练数据
DATA/asset_A_attributes.jsonl(2'')──────┼─→ 5(辅助头监督)
DATA/asset_C_reasoning.jsonl(2'')───────┼─→ 6(推理链目标)
DATA/asset_D_whitelist.txt(2'')─────────┴─→ 仅作 2'' 资产过滤依据留档
   (客户方案: 样本永远全量,白名单不过滤样本;代理集伪 GT 场景例外)

3a outputs/phase5_sft_a/final ──--init-from─→ 3b
3b outputs/phase5_sft_b/final ──--ckpt──→ 4a preds_v0.jsonl
   preds_v0 ─→ 4b sw.json ─--sample-weights+--init-from(3b final)─→
   outputs/phase5_sft_hm/final ──--init-from─→ 5
5  outputs/phase5b_aux/final ──yaml init_from─→ 6
6  outputs/phase5d_cot/final ──--ckpt─→ 4a'(再推理)preds_v15.jsonl
   preds_v15 + labels ─→ 7a outputs/kto_data.jsonl
6  final ──yaml init_from─→ 7b(policy/ref 双 adapter 起点,projector 冻结传递)
7b outputs/phase6_kto/checkpoint-* ─→ 7c swa_final(adapter 平均,
   projector 自动从 6 final 附带)
7c swa_final + 6 final/projector.pt ─→ 8 deliverables/
```
- 所有 `--ckpt/--init-from` 目录 = 上一步的 `final/`(内含
  adapter_model.safetensors + adapter_config.json + projector.pt)。
- **projector.pt 是什么(常见疑问)**: 不是 LoRA —— 是 Vision
  Projector(1.2M 参数)的**全参**权重,太小不值得低秩化,PEFT adapter
  文件装不下它所以单独携带。它是**训练期中间产物**,最终交付时被
  merge 进 vision_merged.pt(Step 8 --projector),不单独交给端侧。
  客户拿到的"LoRA 结果" = deliverables/llm_adapter/(LLM 侧热切换,
  标准 LoRA 语义);视觉侧因 RK1828 只支持 LLM 侧多 LoRA,交付的是
  merge 后完整权重(vision LoRA + projector 均已含在内)→ .rknn。
- **7a 的 preds 必须重新用 6 的 final 推理产出(preds_v15)**,
  不能复用 4a 的 preds_v0(那是 v0 模型的错例,KTO 要修的是 v1.5)。
- 推理/KTO 的帧加载存储自适应(视频文件或客户 WDS,自动识别
  meta.storage),客户数据无需落地 mp4。

### 各训练阶段一览: 数据来源 / 结束判定 / 产出 / 下降防护与回退

> 通用回退协议: 每阶段产物在独立目录,**任何阶段监控集验收不达标 →
> 丢弃该阶段产物,下一阶段 --init-from 直接指上一个达标 final**
> (pipeline.yaml 把该阶段 enabled:false 即自动缝合)。

| 阶段 | 数据来源(前置必须就绪) | 结束判定 | 产出(final/ 内容) | 下降防护 / 回退 |
|---|---|---|---|---|
| 3a warmup | labels.jsonl(Step1)| 固定 1 epoch(+早停)| **仅 projector.pt**(无 adapter)| base 全冻结,无下降面 |
| 3b SFT | labels + 3a final | epochs≤5 上限,**eval_loss 早停(patience 3)+ load_best** | adapter 两件套 + projector.pt | 首次实训,无回退对象;验收 loss_cls 在降 |
| 4b HM 续训 | preds_v0(4a)+ sw.json | epochs≤2 + 早停 + load_best | 同上(phase5_sft_hm)| 类膨胀封顶 1.5×;**验收不达 → 弃 hm,5b 直接 init 3b final** |
| 5 辅助头 | labels 全量 + asset_A(2'',已按白名单过滤)+ 上游 final | epochs≤2 + 早停 + load_best | 两件套 + projector + aux_heads.pt | 无资产样本 aux 自动 -100;验收 RT 无提升 → 查资产覆盖率;**弃之,5d init 上游 final** |
| 6 CoT | labels 全量 + asset_C(同上)+ 5 final | epochs≤2,末 0.5 epoch 强制退火 + 早停 | 两件套 + projector | 无推理链样本自动走生产模式;验收 think 泄漏 =0% 且分类不降;**不达 → 弃,KTO init 上游 final** |
| 7b KTO | kto_data(preds_v15!)+ 6 final | **固定 1 epoch,无自动早停**;checkpoint 每 save_steps 落盘 | checkpoint-* + final(projector 透传)| 离线刹车: 对中间 checkpoint 跑监控集,任一分类指标降>0.5 → 停,LR 减半重跑;**再不行弃 KTO 保留 v1.5** |
| 7c SWA | 7b checkpoint-* | 确定性(逐张量平均)| swa_final(含 projector)| 监控集对比 swa vs kto final,**取优者进导出** |

**监控集(1c)怎么被消费**(易混淆,注意):
- 训练**内**的早停信号 = 验证集 eval_loss(分类 ×4 加权,自动);
- 监控集是**离线阶段闸门**: 每个阶段结束后手动跑
  `python -m training.run_inference --ckpt <该阶段final> --labels monitor.jsonl --out m.jsonl [--shard i/n]`
  → `python -m eval.metrics --pred m.jsonl --gt monitor.jsonl`,
  对照上表"验收"列决定进入下一阶段还是回退。
- 大规模推理(4a/7a 的 10 万级)用 `--shard i/n` 多进程/多机切片,
  各片独立断点续跑,最后 `cat` 合并。

**训练不是必须全链跑**(阶段可选/可跳/可从外部起步):
- 编排器: `configs/pipeline.yaml` 每阶段 `enabled: true/false`,跳过的阶段
  链条自动缝合到最近已启用产出;`python -m training.pipeline --dry-run`
  先看执行计划。
- 任意点起步: 已有外部 checkpoint 设 `start_from_checkpoint`,或任何
  train.py 命令直接 `--init-from <目录>`。
- 最小可用链 = 只跑 3a+3b(基础 SFT);4~7 每级都是可选增益
  (hard mining / 辅助头 / CoT / KTO predictably +0.3~2 分,见 training_plan)。

## Step 0 环境自检

- **目的**: 确认依赖齐全、纯逻辑测试全绿,才允许碰 TPU。
- **输入**: 无(代码库本身)。
- **输出**: 测试通过报告。
- **注意事项**: ① torch 无"TPU 版"——正确栈 = `torch(+cpu 构建)` +
  `torch_xla[tpu]`,两者版本号必须一致(ABI 锁定);② 个别用例需 torch;
  ③ tensorboard 是必装依赖(Trainer 默认 report_to)。
- **实测**: `python3 tests/test_core.py` → **23/23 passed**;
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

## Step 2 Gemini 增强标注(资产 A/C/D)—— 客户数据的第一步作业

> **每步独立执行原则**: 本文档各 Step 均可独立运行,只要满足其
> "前置条件";每步末尾有"验收标准",**达标后才进入下一步的训练实验**。

### 2A 客户 GCS/WDS 数据标注(正式路径,annotation/label_euno_wds.py)

- **目的**: 对 GCS 上的 WDS 数据产出属性(A,喂辅助头)/ 推理链
  (C,喂隐式 CoT);随后 gt 模式过滤产出白名单(D)。
- **前置条件**: Step 1 的 GT 转换已完成(`data/euno_wds.py` 产出
  labels.jsonl);Vertex 项目对 Gemini 有 `aiplatform.endpoints.predict`
  权限;GCS 读权限。
- **输入**: `--wds-dir gs://…/anker_video_clips_wds_full`(或本地)、
  `--annotation` euno 标注 json(限定只标 balanced 100k 覆盖的 key,
  不传会标全量 1M)。
- **命令**(三条,顺序执行):
  ```
  python -m data.euno_wds --annotation <euno标注.json> --wds-dir <dir> --out DATA/labels.jsonl
  python -m annotation.label_euno_wds --wds-dir <dir> --annotation <euno标注.json> \
      --out pass1.jsonl --model gemini-3.1-pro --vertex-project <P> --workers 8 \
      [--shards 0-99]   # 可按分片切分多机并行
  python -m annotation.consistency_filter --mode gt \
      --gemini pass1.jsonl --gt DATA/labels.jsonl --out-dir filtered/
  python -m annotation.split_assets --gemini pass1.jsonl \
      --whitelist filtered/whitelist_ids.txt --out-dir DATA/
  ```
  (第 4 条是训练消费的最后一环 —— 没有它,pass1 的属性/推理链
  嵌在 gemini_output 里,训练会**静默拿不到**辅助头/CoT 监督。)
- **输出**: pass1.jsonl(逐条 attributes/reasoning/predictions)→
  filtered/{whitelist.jsonl, whitelist_ids.txt, partial_match.jsonl,
  discarded.jsonl, gt_suspect_stats.jsonl}。
- **注意事项**: ① 输入是 16 帧图序列(数据已上游采样),labeler 以
  16 个 image part + 时序前言喂 Gemini;② `max_output_tokens` 必须
  8192(thinking 模型思考 token 吃输出预算,900 截断 JSON——真 API
  踩过);③ 断点续跑安全(--out 已有 id 自动跳过),可随时中断;
  ④ 吞吐: 2.5-flash 单条 ~5s,--workers 8 + 分片并行估算 100k ≈
  1~2 天/单机,注意项目配额;⑤ uuid 命名样本 camera_id=unknown,
  转换后对其补跑 `data/camera_fingerprint.py`。
- **首要原则(客户已确认: 人工修正的 GT 质量 > Gemini)**:
  **GT 全量信任 —— 白名单只裁"Gemini 自产资产"(属性/推理链)的
  采用范围,永远不裁训练样本、不裁 GT。** 用 Gemini 是否认同来过滤
  训练样本 = 用低质量裁判筛高质量标签,会系统性丢弃 Gemini 盲区难例。
  落地: split_assets 在资产层过滤;5b/5d 全量样本参训(无资产的样本
  自动降级: aux 标 -100 / CoT 走生产模式);KTO 数据构造不传 --whitelist。
- **GT 来源须知(客户已确认)**: 正式数据 GT = Gemini 标注 + 人工修正。
  两个解读修正: ① 二次 Gemini 与 GT"一致"存在**同源偏置**(两次
  Gemini 可能同错而人工未改到),白名单语义 = "Gemini 增强标注
  (资产 A/C)可信子集",用于 5b/5d 资产质量筛选正确,但**不可当
  GT 对错的独立证据**;② discarded/gt_suspect_stats 的高频不一致对
  = **Gemini 系统性盲区**(报我方改标注 prompt),而非 GT 错标要重标。
- **验收标准(达标才开训)**: whitelist 率 ≥85%(GT 人工修正过,
  预期高于纯机标场景;过低优先怀疑 Gemini 盲区而非 GT);
  .errors 文件占比 <2%;gt_suspect_stats 高频对汇总报我方。
- **实测(2026-07-08,真 API)**: mini-WDS(2 条真实监控帧样本,
  cloud-llm-preview1 / 2.5-flash)全链跑通: 标注 2/2 ok → gt 过滤
  whitelist=1/2 —— 与 GT 一致者进白名单,规则粗标的异常段被 Gemini
  正确质疑丢弃(质量闸门符合设计)。客户环境换 3.1-pro 只改 --model。

### 2B 代理数据集标注(视频文件路径,annotation/gemini_labeler.py)

- 与 2A 同产出,输入为视频文件(我方代理集无 WDS);伪 GT 用
  `--mode double` 双温度一致性替代 gt 模式。
- **实测**: UCSD 98 条 pass1 全部完成 + pass2 31 条截断演示,
  labeler/过滤链真 API 验证通过。

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
- **⚠️ 朴素"错例×3"的整体下滑风险与防护**(必读):
  错例扎堆在难类 → 全量 ×3 隐性改写类别先验,自然分布测试集上常见类
  被挤压,总 ACC 可能净降。防护(hard_mining 已内置):
  ① `--max-class-inflation 1.5`: 每类训练质量膨胀封顶(默认开);
  ② 续训必须带早停 + 监控集(train.py 默认 load_best);
  ③ `--whitelist` **按 GT 来源选用**(客户已确认: 正式数据 GT =
    Gemini 标注 + 人工修正):
    - **客户正式数据 → 不传 --whitelist**: GT 人工背书,错例基本
      都真;Gemini 不认同 GT 的样本恰是"人工看得出、Gemini 看不出"
      的盲区难例,过滤 = 错杀最有价值样本
    - 代理集伪 GT(纯 Gemini,无人工)→ 传,防噪声放大
  推荐命令(客户数据): `python -m training.hard_mining
  --preds preds_v0.jsonl --labels <labels> --out sw.json`
- **验收标准**: 续训后监控集**总指标不降**且热区对(h/n、k/l、C/D)
  混淆减少;class_inflation 表逐类过目。
- **实测**: (演练进行中,待补)

## Step 5 Phase 5b 辅助头(消费资产 A)

- **目的**: 7 属性头挂 Projector 输出,把身份/动作/时长知识蒸进视觉表征。
- **输入**: hm(或 5_b)final + labels 全量 + asset_A(2'' 已按白名单过滤)。
- **输出**: `outputs/phase5b_aux/final`(含 aux_heads.pt,不进交付物)。
- **注意事项**: 样本全量参训(GT 全量信任);无资产/低置信度(<0.5)
  样本的 aux loss 自动置 0(标 -100),不丢样本。
- **实测**: (演练进行中,待补)

## Step 6 Phase 5d 隐式 CoT(消费资产 C)

- **目的**: 推理链蒸馏(60% [REASON] 模式)+ 末段纯生产模式退火,
  防 `<think>` 泄漏。
- **输入**: 5b final + labels 全量 + asset_C(资产层已过滤)。
- **输出**: `outputs/phase5d_cot/final`(v1.5)。
- **注意事项**: 验收必测 think 泄漏率 = 0%(eval/metrics think_leak_rate)。
- **实测**: (演练进行中,待补)

## Step 7 KTO + SWA

- **目的**: 纯分类偏好精修(desirable=GT 串 / undesirable=分类错误输出),
  SWA 平均最后 N 个 checkpoint。
- **输入**: v1.5 final + preds_v15 → kto_data.jsonl(客户数据不传
  --whitelist,desirable=GT 全量可信)。
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

## Step 9 1M 自然分布续训轮(第二轮,configs/phase7_1m_calib.yaml)

- **目的**: 1M 相对 100k 的本质增量是 ~90 万条全新视频(新家庭/机位/
  光照/长尾实例)= 真实表征能力增量(客户历史 100k→1M 提升显著)——
  因此是**真续训**: v2 热启 → SFT ≤2ep(LR×0.5)→ 1M Hard Mining →
  半轮 KTO + SWA;**若 1M 资产就绪则加跑 5b'/5d'**(+2~3 天)。
  资产策略: D1 即对 wds_full 全量启动标注(label_euno_wds 不传
  --annotation=标全量,不需要等 GT;与训练零资源冲突),3~4 天
  @50 并发,一次标注服务两轮(100k 资产=其子集)。
  成本口径: 2.5-flash 约 \$2k 级/1M,3.1-pro 更高 —— 开跑前与
  客户确认预算与 Vertex 并发配额(QPM)。
  排期告急的保底降级: 1ep + LR×0.3,跳 HM/KTO(纯先验校准)。
- **前置条件**: ① v2(swa_final)② **1M 全量 GT 标注文件(客户提供,
  关键路径!现库存仅 balanced 100k 标注)** ③ val_size 调回 30000
  ④ 监控集用 1M 重建(自然分布下安全类配额才真正生效)。
- **命令**:
  ```
  python -m data.euno_wds --annotation <1M标注.json> --wds-dir <wds_full> --out DATA/labels_1m.jsonl
  python -m data.camera_fingerprint --labels DATA/labels_1m.jsonl ...   # unknown 兜底
  python -m eval.monitor_set --labels DATA/labels_1m.jsonl --out monitor_1m.jsonl
  PJRT_DEVICE=TPU python training/train.py --phase configs/phase7_1m_calib.yaml
  ```
- **输出**: outputs/phase7_1m_calib/final → 按 Step 8 重新导出。
- **注意事项**: ① LR×0.3 小步校准,大 LR 会冲掉 100k 轮成果;
  ② 自然分布常见类占大头 → 警惕向多数类漂移压掉安全类;
  ③ v6e-8 全程预算: SFT 3~5 天 + HM 1 天 + KTO/SWA 1~1.5 天≈ 5~7 天(保底降级版 1.5~2.5 天)。
- **验收标准**: 冻结测试集 v2' ≥ v2(总 ACC 应升,主要来自先验对齐);
  安全类 recall ≥ v2−1pt;热区对不回退;格式合规 100%、think 泄漏 0%。
- **回退**: 任一红线破 → 半轮 KTO(1M 错例更丰富)+ SWA 再试;
  仍不达 → 放弃校准,交付 v2(100k 版)。

## 已知与正式训练的差异(演练局限)

| 项 | 演练 | 客户正式 |
|---|---|---|
| 数据 | 98 clips 规则伪标 | 1M 真 GT |
| 资产 A/C/D | 规则模拟 | Gemini 3.1-pro 双温度过滤 |
| 步数 | 0.2~1 epoch | REPRODUCE 默认 + 早停 |
| 硬件 | v6e-1 单芯 | v6e-8(8 核,bs8 显存待首跑确认)|
| 指标意义 | 仅机制验证 | 真实质量指标 |
