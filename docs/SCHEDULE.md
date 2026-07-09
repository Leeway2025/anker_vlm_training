# 2~3 周执行排期(2026-07-09 定稿)

## 双线并行结构
- **标注线(API,与训练零资源冲突)**: D1 对 wds_full 全量启动
  label_euno_wds(不传 --annotation,不等 GT)→ D4~5 完成
  (50 并发口径;开跑前确认 Vertex QPM 与预算: flash 约 $2k 级/1M)。
  一次标注服务两轮(100k 资产 = 其子集)。
- **训练线(TPU v6e-8,bs4+accum8)**:

| 时间 | 主线 | 并行 |
|---|---|---|
| D1~2 | 环境/转换/字节核对/吞吐预检 → Phase 5a+5b 开训 | 标注启动;客户 RKLLM 工具链(tiny LoRA 已随包提供)|
| D3~5 | 5b 早停收敛 + HM(推理→续训)| 标注完成 → gt 过滤 → split_assets |
| D5~8 | 5b_aux → 5d | 基线复现比对(euno_results_adapter)|
| D8~10 | v1.5 推理 → KTO → SWA → 导出评测 → **✅ v2 交付** | 端侧 INT8/延迟用 v2 先行 |
| D10~17 | **1M 续训轮**(phase7): SFT≤2ep → HM → [5b'/5d'] → KTO → SWA | |
| D17~21 | 缓冲: 排障/追加 epochs/灰度准备 | |

## 关键路径与风险
1. **1M 全量 GT 标注文件 —— D7 前必须由客户交付**(现库存仅 balanced
   100k 标注;wds_full 只有帧)。唯一不受我方控制的关键路径。
2. 输入管线吞吐: 开训前跑 `scripts/bench_dataloader.py`,供给不足先
   扩 worker/本地盘,勿让 TPU 空转。
3. spot 纪律(如用 spot): checkpoint 目录放 GCS/gcsfuse;所有作业
   断点续跑,抢占只损失当前 step。
4. 第 50~100 步读稳态步时 → 当天把本排期校准到小时级。
