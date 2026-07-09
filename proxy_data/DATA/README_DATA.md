# 代理数据集(UCSD ped1/ped2 → 客户格式)数据说明

| 文件 | 来源与性质 |
|---|---|
| labels.jsonl | 98 clips,规则伪标(meta.label_source=rule_pseudo): UCSD Train 段→D|m,Test 段→C|s;正式代理训练建议改用 filtered_double 的 Gemini 伪 GT |
| labels_fp.jsonl | camera_fingerprint 自动机位版(98→2 伪机位,与真实机位零混淆) |
| monitor.jsonl | 分层监控集(本数据仅 2 类,自然分布补齐) |
| gemini_pass1.jsonl | **真实 Gemini 标注**(gemini-2.5-flash / Vertex,98/98,temperature 0.1) |
| gemini_pass2.jsonl | 同上 temperature 0.4,31 条(演示性截断,断点续跑可补全) |
| filtered_double/ | 双温度一致性过滤产物(在 pass1∩pass2=31 上运行) |
| asset_A/C | **真实资产**(pass1 拆分,白名单资产层过滤) |
| asset_D_whitelist.txt | filtered_double 白名单 |
| *.errors | 标注失败记录(重跑会自动重试这些 id) |

注意: 客户正式数据(euno WDS)对应流程见 docs/WALKTHROUGH.md Step 2A,
GT 已含人工修正,过滤走 --mode gt 且样本永远全量参训。

> 白名单率 32/98 是 pass2 演示性截断(31 条)的产物 —— 交集外样本
> 自动进 discarded,并非质量信号;补全 pass2 后按 ≥80% 口径验收。
