# Anker VLM 训练代码库(Gemma 4 E2B / TPU v6e)

> 交付给客户的训练代码。客户用自己的数据(当前 ~10 万,规划 1M)重训;
> 我方在代理数据集上完成代码验证(数据不出客户环境)。
> 方案依据: `../training_plan.md`(定稿版)。

## 主路线:JAX(训练/推理/评测/导出全流程)

**训练与评测一律走 `jax_impl/`**——吞吐 ≈ torch_xla 的 1.9×(8 芯 100k
约 1.4h/epoch),编译 2 分钟(torch_xla 15~30 分钟),全流程已在客户
v6e-8 投产。

```bash
# 快速开始(TPU VM;镜像=纯环境件,代码=git commit,两者分离)
docker pull europe-west4-docker.pkg.dev/leeway-main/anker/jax:env-v1
git clone https://github.com/Leeway2025/anker_vlm_training.git && cd anker_vlm_training
# 逐阶段照做: jax_impl/USAGE.md(S0 环境 → S1/S2 训练 → … → S8 评测 → S9 导出)
```

| 入口 | 内容 |
|---|---|
| [`jax_impl/USAGE.md`](jax_impl/USAGE.md) | **逐阶段命令手册(唯一权威用法)**,含容器包装模板、与 torch 成果衔接 §A/§B/§C |
| [`jax_impl/README.md`](jax_impl/README.md) | 路线状态、版本口径、三条硬规矩 |
| [`jax_impl/FINDINGS.md`](jax_impl/FINDINGS.md) | 设计决策、验证记录、踩坑档案 |

三条硬规矩(细节见 jax_impl/README):推理评测必须 `infer_sharded.sh`
全芯并行;评测/交付用 `train_params_best.npz`;gemma 库版本已 pin 勿动。

## 路线分工

| 环节 | 用哪条路线 |
|---|---|
| S1~S7 训练(SFT/难例/辅助头/CoT/KTO/SWA) | **JAX**(`jax_impl/`) |
| 推理、分类指标评测 | **JAX**(`infer_sharded.sh` + `eval_metrics.py`) |
| 训练成果导出为 HF peft adapter(含视觉塔) | **JAX**(`export_hf.py --scheme prod`) |
| Gemini 标注、数据资产 A/C/D、euno-wds 转换 | torch 侧工具(`annotation/`,`data/euno_wds.py`) |
| hf_layout.json 生成(一次性,改 prompt 才重跑) | torch venv(`jax_impl/poc/02a`) |
| RKLLM 端侧交付(拆分/合并/onnx/Issue #480) | torch 侧(`export/`,`docs/issue480_workaround.py`) |
| 回退兜底 | torch 全流程保留可用(下节) |

adapter 在 HF peft 格式层双向互换,任何时刻可整体退回 torch 路线
(数据、格式、超参三层兼容)。

## 结构

```
jax_impl/      ★ 主路线: JAX 训练/推理/评测/导出(独立实现,零 torch 依赖)
configs/       超参与生产 prompt(两路线共用口径;prompt.txt 分隔符与客户 GT 字节核对)
data/          taxonomy(类别单一来源)/ formatting / sampling / euno_wds(两路线共用语义)
annotation/    gemini_labeler(3.1 Pro 标注)/ consistency_filter(白名单)
training/      torch 路线训练代码(Phase 5/5b/5d/KTO/SWA;支撑与回退)
eval/          torch 侧评测工具(metrics/format_validator/monitor_set)
export/        RKLLM 端侧交付链(split_deliverables + export_onnx)
docs/          REPRODUCE.md(torch 客户手册)/ issue480_workaround.py
tests/         纯逻辑单测(python3 tests/test_core.py,无 torch 依赖)
```

---

## torch 路线(支撑与回退;以下为其专属细节)

手动逐阶段(与 training_plan Phase 对应)或编排器执行:

```
编排器: 编辑 configs/pipeline.yaml(阶段开关)→ python -m training.pipeline --dry-run → 执行
手动:   annotation → train.py --phase phase5_sft.yaml --stage a/b → phase5b_aux
        → phase5d_cot → build_kto_data → kto.py → swa.py
        → split_deliverables.py [--issue480] → export_onnx.py → eval/metrics.py
```

### TPU 烟测清单(2026-07-07 v6e-1 真机 9/9 PASS)

`PJRT_DEVICE=TPU python tests/smoke_tpu.py` 可随时回归。要点:模型类
`Gemma4ForConditionalGeneration`;processor 必须 `do_sample_frames=False`
(内置采样器默认重采 32 帧);70 token/帧 × 16 = 1120;PLE 实名
`per_layer_*`;`layer_types` global = [4,9,14,19,24,29,34];PISSA/
rank_pattern 512/256 生效;kto.py 全链路真机跑通。真机修掉的坑详见
WORK_STATUS 第 4 轮。

### 红线(代码级强制,两路线同口径)

- base/PLE/Embedding 冻结断言(common.freeze_base 未命中即 raise)
- 时序翻转禁止(augmentation.assert_monotonic,违者抛异常)
- 输出解析按位置取字段 + 大小写矫正 + RT×SubKS 合法组合校验
  (非法组合表含 D|q,涉安全类升级为 C 告警,training_plan 14.2)

### 与 training_plan 的已知偏离(交付评审需知)

**KTO 实现**: 方案 10.2 写 TRL KTOTrainer,实际为自实现(training/kto.py
与 jax_impl/kto.py,损失数学与 TRL 逐项对齐)。决策依据(2026-07 调研):
TRL KTOTrainer 无 TPU 支持且不支持视频输入;EasyDeL 纯文本管线偏离路线;
本实现 ref 设计与 TRL v1.x 官方机制一致,tests/test_kto_math.py 数值对拍。
Plan B(客户可用 GPU 时): TRL v1.6+ 多图支持可把 16 帧当图像序列,仅作兜底。

### Phase ↔ 代码映射(training_plan.md 对照)

| Phase | torch 侧 | JAX 侧(主) |
|---|---|---|
| 0 验证准备 | eval/metrics.py / issue480 / tests | — |
| 1 数据准备 | build_dataset.split_by_camera / monitor_set | jax_impl/data.py(同语义切分内建) |
| 2 Gemini 标注 | annotation/*(资产 A/C/D) | (复用 torch 侧产物) |
| 3 采样 | data/sampling.py + augmentation | jax_impl/data.py(euno-wds 直读) |
| 4 LoRA | common.freeze_base/build_lora | jax_impl/prod_lora.py(同 rank 方案) |
| 5 基础 SFT | train.py --stage a/b | **train_sft.py --stage a/b**(S1/S2) |
| 5 难例 | hard_mining.py | **infer_sharded + hard_mining.py**(S3) |
| 5b 辅助头 | phase5b_aux + AuxHeads | **--aux-file/--ks-head**(S4) |
| 5d 隐式 CoT | phase5d_cot + build_cot_target | **--cot-file/--cot-anneal**(S5) |
| 6 KTO | build_kto_data → kto.py | **jax_impl/kto.py**(S6) |
| 9 SWA | training/swa.py | **jax_impl/swa.py**(S7) |
| 11 导出 | split_deliverables → onnx | **export_hf.py --scheme prod**(S9)→ 接 torch 端侧链 |
| 监控/评测 | eval/metrics.py | **eval_metrics.py**(S8,同口径) |
