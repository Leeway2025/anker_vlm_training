# 1M 数据扩展作战计划(2026-08-01 定稿)

**目标**:用 1M 池兑现大 rank 配方的设计容量(客户基准 E2B 100k→1M = +6.5),
冲击 SubKS 76+ / RT 85+。**口径铁律**:与 EunoVLM 同池同法——不剔测试集重叠、
不做复制式配平,一切与 100k 阶段同规则。

## 资源分工(三线并行,互不抢)

| 资源 | 承担 |
|---|---|
| CPU(任意机) | P0 盘点 / P1 体检 / P3 策展 |
| 网络(Gemini) | P2 盲判抽样(与 RT 重标共享配额,排队) |
| **GPU 机** | **重切帧 24~32 + WDS 重打包(关键路径最长杆,尽早开工)** |
| TPU v6e-8 | P5 冒烟起的所有训练 |

## P0 盘点(半天,问话+ls)

向数据方确认四样:
1. 1M labels 文件路径与行数;
2. 1M WDS 分片目录、总大小、index.json 是否齐;
3. 标注血统:哪批人/什么工具/是否做过配平或翻译;
4. GPU 机档期(重切帧用)与 Gemini 配额(P2 要 ~2 万条)。

## P1 全池体检(纯 CPU,分钟级)

```bash
bash scripts/profile_pool.sh /path/to/labels_1m_raw.jsonl
```
产出:规模/去重/测试集重叠(仅记账)/RT×SK 分布 vs 测试集(差>5点=采样嫌疑)/
懒标浓度(对照 100k 水位 25.4%)/desc 质量(中文比例→决定要不要翻译轮)/机位集中度。

## P2 分层盲判抽样(网络,~1天)

```bash
export GEMINI_API_KEY=AQ.xxx  WDS_1M=<1M WDS目录>
nohup bash scripts/pool_sample_blind.sh /path/labels_1m_raw.jsonl > pool_blind.log 2>&1 &
# 跑完出分层错标率地图:
bash scripts/pool_blind_report.sh /path/labels_1m_raw.jsonl
```
判读基准:一致率 ~37% 是盲判正常水位;显著低于 100k 同类水位的类 = 标注恶化信号。
**纪律:不做整类清洗,只人工抽查;懒标但与考卷同源者(m/D)一律保留。**

## P3 策展出池(CPU,小时级)

```bash
bash scripts/build_pool_v1.sh /path/labels_1m_raw.jsonl
# 产出 /data/labels_1m_v1.jsonl + /data/val_ids_1m.txt(1536条 test-mix 专属 val 卷)
```
规则:去重 + 坏行/空 desc 剔除 + 中文 desc 暂出池;自然分布保留(不配平);
不剔测试集重叠。血统记入 logs/README。

**P3.5(条件触发)翻译轮**:若 P1 显示中文 desc 占比 >5%,起 Gemini 批量翻译
(复用 rationalize 并发骨架),译完并回池。100k 时代 no_trans 的教训:不译不能用。

## P4 工程准备(与 P1-P3 并行)

- **GPU 重切帧(尽早)**:24~32 帧/条(B 桶归因:o/j/s 等"证据掉在帧间"的类是
  受益者,可赎回 ~2 分);建议 NVDEC 硬解;切帧参数定稿前先拿 200 条切样验证
  (均匀采样 vs 关键帧,分辨率与现管线一致);
- WDS 覆盖与吞吐验收:
```bash
bash scripts/check_wds_pool.sh /data/labels_1m_v1.jsonl <新WDS目录>
```
- **训练断点续训**(train_sft checkpoint+optimizer 状态落盘,长跑刚需):
  周二实现并在 TPU 验证——**上线前 train_1m.sh 不得在会关机的窗口启动**;
- 机时窗口:全量一发 25~35h,与机器关停策略必须先对齐。

## P5 管线冒烟(过夜,TPU)

```bash
nohup bash scripts/smoke_1m.sh > smoke_1m.log 2>&1 &
```
从 v1 池自然抽 10 万,v4 配方逐字训一发,对照 seed-1 裸分 73.52:
±0.5 内 = 新池质量同级,放行全量;低 1 分以上 = 池有系统性问题,回 P2 地图找病类。

## 全量首发(P5 放行后)

```bash
nohup bash scripts/train_1m.sh > train_1m.log 2>&1 &   # 草案: 8000步/eval250/patience4
```
出炉后依次叠加 100k 时代已验证的零成本增益:手术先验+RT 阈值(两折重标定)→
种子摇号(机时允许)→ logits 集成。

## 决策门与风险

| 门 | 条件 | 动作 |
|---|---|---|
| 翻译轮 | 中文 desc >5% | P3.5 起翻译,预算另计 |
| 冒烟门 | smoke vs 73.52 差距 >1 | 暂停全量,回归因 |
| 机时门 | 断点续训未上线 | 全量不启动 |
| 配方门 | 8000 步草案 | 冒烟的 loss/val 曲线定稿最终步数 |

## 已备工具清单

profile_pool.sh(P1)/ pool_sample_blind.sh + pool_blind_report.sh(P2)/
build_pool_v1.sh(P3)/ check_wds_pool.sh(P4)/ smoke_1m.sh(P5)/ train_1m.sh(全量)
——全部一键,git pull 即得。
