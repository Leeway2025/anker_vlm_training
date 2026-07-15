# JAX 路线(官方 google-deepmind/gemma 库)—— 平行验证实现

**与现有 torch_xla 路线完全隔离**:本目录不 import `training/`/`data/` 的任何
torch 代码(仅 PoC-02a 例外,它就是要在 torch 栈里跑出基准),不改动任何
现有文件;独立 venv(`setup_jax_env.sh`),现有训练不受影响。

## 为什么值得验证(2026-07-15 调研结论)

- 官方 JAX 库已内置 `gm.nn.Gemma4_E2B` + `CheckpointPath.GEMMA4_E2B_IT`,
  含 vision/(ViT,3×3 池化,max_soft_tokens=1120,与生产同参)与 audio/。
- 内置 LoRA(`gm/nn/_lora.py` + 官方 colab)。
- 预期训练提速 1.3~1.8×(细粒度 remat 替代全量重算、步内零 host 同步、
  无 HF Trainer 开销),epoch 7~10h → 4~6h。
- **两个未决风险,必须先验证**:①视频语义——JAX 预处理只有独立图像,
  没有视频/时间维,16 帧需按 16 图拼装,token 排布必须与 HF/生产端逐
  token 对齐(RKLLM 端侧按 HF 语义跑,错位=训练/部署脱节);
  ②LoRA → HF peft 格式导出(RKLLM 只吃 HF 格式)。

## 验证门槛(按序执行,任一 NO-GO 即停,损失可控)

| Gate | 脚本 | 验证什么 | NO-GO 含义 |
|---|---|---|---|
| A | `poc/01_load_model.py` | E2B 类可实例化、checkpoint 可加载、参数量≈5B、视觉参数树存在 | 库/权重不可用,路线终止 |
| B | `poc/02a_dump_hf_layout.py` + `poc/02b_jax_layout.py` | **决定性门槛**:同一 16 帧+生产 prompt,HF 与 JAX 的 token 排布逐位对齐(soft token 数/帧、位置、前后特殊 token、文本段 ids) | 排布对不齐且无法通过拼装参数修正 → 路线终止 |
| C | `poc/03_forward_parity.py` | 同输入下 HF 与 JAX logits 数值一致(先纯文本,后多模态) | 权重转换/实现有语义差,须查明才可继续 |
| D | `poc/04_lora_export.py` | JAX LoRA 参数 → HF peft safetensors → HF 加载回读数值一致 | 导出链不通,RKLLM 部署闭不了环 |

**全部 Gate 通过后**才开始训练移植(阶段 2):
euno_wds 读取(纯 python,无 torch 依赖,可直接复用)→ grain/生成器数据管线
→ 加权 CE(分类 token ×4)→ LoRA(分层 rank)→ `lax.scan` 梯度累积
→ 选择性 remat 策略 → checkpoint/导出。KS 头/辅助头/KTO 等可选阶段后置。

## 运行方法(v6e 机器)

```bash
bash jax_impl/setup_jax_env.sh          # 建 /dev/shm/venv_jax(独立环境)
# Gate A(CPU 即可,不抢正在训练的 TPU;上 TPU 时去掉 JAX_PLATFORMS)
JAX_PLATFORMS=cpu /dev/shm/venv_jax/bin/python jax_impl/poc/01_load_model.py
# Gate B:先用现有 torch venv 跑 02a 出基准 JSON,再用 jax venv 跑 02b 比对
<torch_venv>/bin/python jax_impl/poc/02a_dump_hf_layout.py --out /tmp/hf_layout.json
/dev/shm/venv_jax/bin/python jax_impl/poc/02b_jax_layout.py --ref /tmp/hf_layout.json
```

注意:PoC 脚本对官方库 API 做了防御性内省(接口对不上时打印候选属性再
退出),首轮跑通常需要按打印结果小改——这是探路脚本的预期用法。
