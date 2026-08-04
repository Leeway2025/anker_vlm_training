# 数据噪声治理实战手册(Data-Centric AI Playbook)

> Anker VLM 项目 · 2026-08-04 · 基于业界方法综述 + 本项目全部实测数据
> 配套 PDF:docs/data_centric_noise_playbook.pdf

## 0. 摘要

大模型时代的共识:**模型架构的边际收益递减,数据质量的边际收益仍然陡峭**(data-centric AI)。
本项目的标签噪声治理已覆盖业界六大方法族中的全部可用项,核心结论:

1. 我们的组合拳(描述 judge + 多证人合议 + 只升级纪律 + 删改分层 + 评测双口径)与业界 SOTA 实践一一对应,方向无需调整;
2. 新补齐三件套:训练动力学日志(`--cartography`)、嫌疑软化通道(loss 降权)、置信学习交叉验证;
3. 反直觉铁律(实测):清训练集会让冻结 test 的 RT 分**不升反降**(共享错标幻觉),清标验收只看 SubKS/人工,永远不看 test RT。

## 1. 病情画像

| 维度 | 数字 | 来源 |
|---|---|---|
| 训练集 | 83,561 条(去重后) | labels_dedup.jsonl |
| 测试集 | 11,022 条,**冻结、客户已验、永不清洗** | labels_test.jsonl |
| RT 训练集错标(desc-judge) | 4,240 = 5.1%(D 3061 / A 1075 / B 90 / E 14) | gt_desc_judge_train.jsonl |
| RT 测试集错标(desc-judge) | 522 = 4.7%(D 8.5% > A 5.9% > E 0.8% > B 0.3% > C 0%) | test_mislabel_exclude_ids_522.txt |
| SubKS 测试集错标(desc-judge, 修正后) | 622 = 5.6%(e→b 424 系统性遛狗错标) | test_sk_mislabel_exclude_ids.txt |
| SubKS 两证人嫌疑上限 | 882 = 8.0%(含主观边界,不可直接当错标) | 模型×盲判交叉 |
| 本质模糊(全片都判不了) | RT 错误的 49% / SubKS 错误的 66% | 16帧盲判分解 |
| 关键帧缺失真实上限 | RT 1.48pp / SubKS 4.01pp(事件类双双漏判) | 同上 |

**噪声结构三分**:①客观错标(描述自打脸,可清)②主观边界(A↔D、e/f/m、k/l,清不动,只能软化)③瞬时事件漏标(帧间证据,需客户侧重切,收益上限小)。

## 2. 方法族总览

| # | 方法族 | 代表工作 | 本项目状态 |
|---|---|---|---|
| 1 | 置信学习 | Cleanlab / Northcutt 2021 | ✅ 已跑(CL 615 条,∩judge 72 = 证人互补)|
| 2 | LLM-as-Judge 清洗 | FineWeb, Nemotron-CC | ✅ 主力(RT 4240 / SK 827→622)|
| 3 | 训练动力学 | Dataset Cartography / AUM | ✅ 已落码(--cartography),清洗夜启用 |
| 4 | 噪声鲁棒训练 | label smoothing / co-teaching / DivideMix | ✅ 软化通道已落码(降权 0.3)|
| 5 | 主观边界正规化 | learning from disagreement | ✅ 纪律已建(只升级/不判 k-l-m)|
| 6 | 测试集卫生 | ImageNet-ReaL / CIFAR-N | ✅ 双口径已接管线 |

## 3. 置信学习(Confident Learning)

**原理**:对每类 j 估计"被标为 j 的样本的平均预测概率"作为类阈值 t_j;样本(标 i)若对某类 j≠i 的预测概率 ≥ t_j,计入混淆联合 C[i][j] 的 off-diagonal → 嫌疑。比裸 argmax 稳健(类不平衡下阈值自适应)。

**经典结论**(Pervasive Label Errors, Northcutt 2021):ImageNet test 有 ~6% 错标;**模型会背测试集错标刷分,测试集越脏、大模型优势越被低估**。本项目"剔 522 后 RT +1.26pp"就是该现象的复现。

**本项目实测**(S5 模型 RT 字母 logits,test 11022):
- CL 嫌疑 615 = 5.6%;方向 A→D 236 / D→A 141(主观边界为主)
- **与 judge 522 只重合 72 条** —— 两个证人盲区互补:CL 看不见"模型已背下的共享错标"(模型附和错 GT 时 CL 无信号);judge 看不见"描述没写出来的错标"
- ⚠️ CL 清单(cl_rt_suspects_test.json)**只能用于训练集清洗排序/人工审计,绝不可当评测剔除口径**(它使用了被测模型,不满足 model-independent)

**落地**:清洗夜产出新模型后,用其 train 集 logits 跑 CL,与 desc-judge 4240 取交集 = 最高置信错标(优先修),取差集 = 各自盲区(送人工抽审)。

## 4. LLM-as-Judge 清洗(主力方法)

**设计要素**(业界共识 ↔ 本项目实现):
1. 只抓肯定性矛盾,模糊一律 ok("vague → ok")—— 宁漏抓保精度
2. 要求引用证据(cue 字段 ≤12 词原文引用)—— rationale-required,防幻觉
3. prompt 约束 + **代码硬闸双保险**(ALLOWED_TARGETS / NEVER_DEMOTE / 身份闸)
4. 试跑(分层抽样)→ 人工目测 → 收紧 → 全量,迭代闭环
5. 删除比改判安全:三层口径 = 有旁证才修(1933)/ 单证只删或降权(2307)

**两次实战教训**:
- 第一轮试跑暴露"多主体场景"与"干活途中碰车"两个误杀模式 → prompt 加 MAIN-activity 约束,嫌疑率 4.1%→3.0%
- SubKS 全量后审计发现 m→p 系统性误杀 277 条:**prompt 类定义没对齐父类语义**(p=野生动物,父类 LifeThreat;家猫家狗标 E|m 本来就对)→ 后过滤只留明确野生物种 70 条。教训:**judge prompt 的类定义必须逐条对照权威 taxonomy 与父类分组写**。
- e→b 424 条(e 类的 46.5%!)是真·系统性错标:标注员把遛狗成批标进"休闲",94% 有旁证 —— 系统性错标一旦命中,单方向即可达全类近半,这是 judge 相对逐条人工审的核心价值。

## 5. 训练动力学(Dataset Cartography / AUM)

**原理**:训练过程中每样本每次相遇的 loss/margin 轨迹,天然把数据分三桶:
- **易学**(低 loss 低方差)= 干净样本
- **难学但稳定**(高 loss 低方差)= 真难例(保留!是泛化的养分)
- **高波动/低置信**(高方差)= 疑似错标

这是**唯一能区分"错标 vs 真难例"的免费证人**——desc-judge 和 CL 都做不到这一点。

**本项目实现**(train_sft.py `--cartography`,2026-08-04):
- loss_fn 借 `has_aux` 返回每样本加权 CE,零额外前向;宿主用 draw() 纯函数反推 video_id,预取器零改动
- 输出 `<out>/cartography.jsonl`:{video_id, micro, loss},~4.7 epoch × 83k ≈ 39 万行
- 分析:按 video_id 聚合 → 均值(难度)× 方差(波动)二维图 → 高方差桶 ∩ judge/CL 嫌疑 = 铁证错标;高均值低方差 ∩ 嫌疑 = 大概率真难例,从清洗清单摘除

**用法**:清洗夜训练命令加 `--cartography`,一次训练白捡第四证人。

## 6. 噪声鲁棒训练(不清洗,让模型自己扛)

业界谱系:label smoothing(已用)→ 广义交叉熵 GCE → co-teaching(双模型互挑低 loss 样本)→ DivideMix(loss 分布 GMM 软分干净/噪声池 + 半监督)。后两者工程重,VLM SFT 场景性价比低。

**本项目落地 = 软化通道**(2026-08-04):
- `sample_weights` 双语义:w>1 物理复制(hard-mining 原有)/ **0<w<1 → 该样本整条 loss 降权**(data.py 乘进 per-token weights,仅 train 生效,val 不动;已单测)
- 资产:`labels_dedup_softclean.jsonl`(83,561 全保留:1933 修正 + 2307 降权 0.3)+ `suspect_weights.json`
- **软化 vs 硬删的哲学**:单证嫌疑 precision 有限(~50-70%),硬删会错杀真难例;降权 0.3 = "可疑标签只发 30% 的声音",保信息量、控噪声,期望损失严格优于二选一赌博
- 清洗夜建议跑软化臂(对照:原始 / 硬删 81254 / 软化 83561)

## 7. 主观边界正规化(清不动的,改造它)

**业界结论**(learning from disagreement / CrowdTruth):标注员分歧是**信号不是噪声**;对本质模糊的边界,强行仲裁 = 制造伪确定性。

本项目对应纪律(全部已建):
- A↔D:无正面证据不判(D 是兜底不是"陌生人");**76% 本质不可判**,不清、不调模型,弹药押 SubKS
- C:只升级不降级(描述/帧没写威胁 ≠ 视频无威胁;Gemini 读不出犯罪意图是它的盲区)
- k/l/m:文本方向不可靠,judge 互不改判
- 层级兜底:KS 父类 acc(6 类)对边界混淆免疫,是更稳的汇报口径
- 后续可选:对边界桶用软标签(A/D 各 0.5)训练,把"不确定"显式教给模型

## 8. 测试集卫生(脏尺子怎么用)

**铁律:测试集只读诊断,永不清洗**(冻结基准 = 客户验收的尺子;改它 = 分数不可比 + 越权)。

**共享错标幻觉**(本项目实测,反直觉但已三口径验证):
- 522 条测试错标里,模型附和错 GT 301 条 vs 模型判对被冤 100 条 = **3:1** —— 模型从训练集背下了同一套错标映射
- 三口径对比(单模):全量 RT 83.37 / 剔 522 → 84.65(+1.3)/ **把 522 改成正确标签 → 81.55(−1.8)**
- 推论:清训练集 → 模型学对 → 在 test 的 301 条共享错标上"答对被判错" → **test RT 下降但真实质量上升**。所以清标验收看 SubKS + 人工,永远不看 test RT。

**双口径管线**(已接入 eval_metrics.py + night_chain.sh):官方全量(11022)永远第一、gating 只认它;去噪口径(RT 剔 522 / SubKS 剔 622)只做附加诊断。剔除口径必须 model-independent(judge/盲判都不看被测模型)。

**Platinum 子集**(业界:ImageNet-ReaL/CIFAR-N 的路):bucket ③ 高嫌疑 115 条送人工终审,可产出"白金子集"作为第三报告口径。

## 9. 多证人合议架构(本项目的核心方法论)

| 证人 | 看什么 | 盲区 |
|---|---|---|
| desc-judge(Gemini 纯文本) | GT 描述 ↔ 标签自洽 | 描述没写出来的错标;类定义写错则系统性误杀 |
| Gemini 16帧盲判 | 帧 ↔ 标签 | 读不出犯罪意图(C 类);与模型共享帧局限 |
| 我方模型预测 / CL | 训练分布 ↔ 标签 | 已背下的共享错标(与噪声同源)|
| 训练动力学 | loss 轨迹形状 | 需要一次完整训练;重复样本稀释信号 |
| 人工终审 | 一切 | 贵,只审高嫌疑交集 |

**决策规则**:双证人同向 → 修正;单证 → 降权(软化)或删除;证人矛盾 → 人工;涉 C/安全类 → 只升级。
**评测剔除口径的唯一合法证人组合**:desc-judge + 盲判(均不看被测模型)。

## 10. 行动手册(下一步)

1. **清洗夜实验矩阵**(单变量纪律,每臂只动一处):
   - 臂 A:原始 labels_dedup(基线复跑,带 --cartography)
   - 臂 B:软化 softclean + suspect_weights(推荐主臂)
   - 臂 C(可选):硬删 clean 81254
   - 全臂统一:--ckpt-every 200 --resume 断点保护;验收 = SubKS ≥ 基线 & 安全召回不降;**不看 test RT**
2. cartography 出图 → 高方差桶 ∩ 嫌疑清单交叉 → 迭代第二轮清洗
3. 增强 v2 消融(S2b2 门禁)可与清洗夜合并排期:clean 数据 × (v2 开/关)
4. SubKS e→b 424 条系统性错标 → 训练集同方向 desc-judge(预计放大到 ~3000+ 条),纳入下轮清洗
5. bucket ③ 115 条 + CL∩judge 差集抽样 → 人工终审 → platinum 子集

## 参考文献

- Northcutt et al., *Confident Learning: Estimating Uncertainty in Dataset Labels*, JAIR 2021
- Northcutt et al., *Pervasive Label Errors in Test Sets Destabilize ML Benchmarks*, NeurIPS 2021
- Swayamdipta et al., *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics*, EMNLP 2020
- Pleiss et al., *Identifying Mislabeled Data using the Area Under the Margin Ranking*, NeurIPS 2020
- Han et al., *Co-teaching: Robust Training of DNNs with Extremely Noisy Labels*, NeurIPS 2018
- Li et al., *DivideMix: Learning with Noisy Labels as Semi-supervised Learning*, ICLR 2020
- Beyer et al., *Are we done with ImageNet?*(ImageNet-ReaL), 2020
- Wei et al., *Learning with Noisy Labels Revisited*(CIFAR-N), ICLR 2022
- Ratner et al., *Snorkel: Rapid Training Data Creation with Weak Supervision*, VLDB 2017
- Penedo et al., *The FineWeb Datasets*, NeurIPS 2024
