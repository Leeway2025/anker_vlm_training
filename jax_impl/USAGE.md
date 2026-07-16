# JAX 训练路线使用手册

面向已在用 torch 路线的用户。JAX 路线与 torch 路线**互操作**(adapter 在
HF peft 格式层双向互换),可整体迁移,也可只把最耗时的主训搬过来
(混合管线)。实测吞吐:v6e 8 芯 100k 样本 **≈1.9h/epoch**(torch 现状
6~8h);编译 ~2 分钟(torch_xla 15~30 分钟)。

---

## 0. 一次性准备

```bash
# 1) 独立环境(python>=3.12;与 torch venv 完全隔离,互不影响)
bash jax_impl/setup_jax_env.sh /path/to/venv_jax
# 机器只有 py3.10 时用 uv,见脚本头部注释;小系统盘务必挪 UV 缓存目录

# 2) 生成排布模板(一次性;用你们现有的 torch venv 跑,产物给 JAX 用)
<torch_venv>/bin/python jax_impl/poc/02a_dump_hf_layout.py --out hf_layout.json
```

- 权重自动从 `gs://gemma-data/checkpoints/gemma4-e2b-it` 拉取(GCE 默认
  服务账号可读),无需手动下载;
- 数据与 torch 路线**完全同源**:同一份 `labels.jsonl` + euno-wds 分片,
  无需任何转换;
- `hf_layout.json` 是"生产 prompt + 16 帧视频"的 token 排布基准,JAX 侧
  据此保证与 HF/RKLLM 逐 token 一致(改 prompt 后须重新生成)。

以下命令均用 `<venv_jax>/bin/python`,工作目录 = 仓库根。

---

## 1. 训练各阶段

### stage a(仅 projector 预热)
```bash
python jax_impl/train_sft.py \
  --labels DATA/labels.jsonl --layout hf_layout.json \
  --stage a --steps <N> --accum 4 --out outputs/jax_stage_a
```

### stage b(主训 —— 生产配置)
```bash
python jax_impl/train_sft.py \
  --labels DATA/labels.jsonl --layout hf_layout.json \
  --rank-scheme prod --train-vision --train-projector \
  --init-npz outputs/jax_stage_a/train_params.npz \
  --steps <N> --accum 4 --eval-every 50 --out outputs/jax_stage_b
```
`--rank-scheme prod` = 与 torch 生产完全同款的超参:差异化 rank
(全局注意力层 512 / 滑动层与视觉 256)+ rsLoRA(α=2r)。

### hard mining(推理 → 挖掘 → 加权续训)
```bash
python jax_impl/infer.py --labels DATA/labels.jsonl --layout hf_layout.json \
  --init-npz outputs/jax_stage_b/train_params.npz --out outputs/preds.jsonl
python jax_impl/hard_mining.py --preds outputs/preds.jsonl \
  --labels DATA/labels.jsonl --out outputs/sw.json
python jax_impl/train_sft.py ...(同 stage b)... \
  --sample-weights outputs/sw.json \
  --init-npz outputs/jax_stage_b/train_params.npz --out outputs/jax_hm
```
大规模推理也可走 torch 侧多芯分片(先 `export_hf.py` 导出,再
`bash scripts/infer_sharded.sh`,见第 3 节混合管线)。

### 辅助头 / 隐式 CoT(需 Gemini 资产 A/C)
```bash
# 在 stage b 命令基础上追加:
  --aux-file DATA/asset_A_attributes.jsonl --ks-head        # 7 属性头+KS 父类头
  --cot-file DATA/asset_C_reasoning.jsonl --cot-anneal 0.5  # 隐式 CoT+退火
```

### KTO
```bash
python jax_impl/kto.py --kto-data outputs/kto_data.jsonl \
  --labels DATA/labels.jsonl --layout hf_layout.json \
  --init-npz outputs/jax_hm/train_params.npz --out outputs/jax_kto
```

### SWA
```bash
python jax_impl/swa.py --ckpts out1/train_params.npz out2/... --out outputs/jax_swa
```

### 评测
```bash
python jax_impl/infer.py ... --out outputs/eval_preds.jsonl
python jax_impl/eval_metrics.py --preds outputs/eval_preds.jsonl \
  --labels DATA/test_labels.jsonl --per-class
# 输出: RT/SubKS/双对 acc、KS 父类 acc、安全关键召回、格式合规率
```

---

## 2. 与已训练的 torch stage a/b 结果衔接(重点)

### 2.1 衔接 stage b 的 LoRA adapter(可用,已数值验证)

```bash
# torch 的 outputs/phase5_sft_b/final(含 adapter_model.safetensors)→ JAX
python jax_impl/import_hf.py --adapter outputs/phase5_sft_b/final \
  --out outputs/imported_b.npz
python jax_impl/train_sft.py \
  --labels ... --layout hf_layout.json \
  --init-npz outputs/imported_b.npz \
  --train-vision --train-projector --lr <小 LR> --out outputs/jax_cont
```

必须知道的三件事:
1. **不要加 `--rank-scheme prod`**。torch adapter 里 k/v 与 gate/up 的
   LoRA A 矩阵是独立训练的,而 JAX 融合算子共享 A —— import 用
   "rank 加倍拼接"精确表示(数学无损,已用真实 checkpoint 对拍 PASS),
   代价是容器 rank=1024,与 prod 方案(512/256)结构不同,**两种方案
   不可混用**;
2. **前 50~100 步用小 LR 预热**(如 1e-5):Adam 动量不跨框架迁移,
   需要重新积累;
3. 续训后如需回到 torch/RKLLM,用默认 `export_hf.py`(不带 --scheme
   prod)导出;rank=1024 的 adapter 对 RKLLM 的 LoRA 内存预算是否
   可接受需确认,必要时做 SVD 收缩(工具在排期)。

### 2.2 衔接 stage a 的 projector.pt(暂缺转换器,给两个替代)

torch 的 `projector.pt` → JAX 参数树的转换小工具**尚未提供**(在排期)。
当前建议二选一:
- **推荐**:JAX 侧直接重跑 stage a —— projector 只有 1.2M 参数,
  1 epoch 在 8 芯上约分钟级,重训成本可忽略;
- 或保持 torch 路线跑完 stage a+b,只在 b 之后用 2.1 的方式接入 JAX。

### 2.3 反方向:JAX 训练结果 → torch 生态(推荐的混合管线)

```bash
python jax_impl/export_hf.py --npz outputs/jax_stage_b/train_params.npz \
  --out outputs/jax_adapter_hf --scheme prod
```
产物与 torch 训练出的 adapter **同格式同配置**(rank_pattern/
alpha_pattern/use_rslora),torch 侧的 run_inference、评测、RKLLM 导出
链全部直接可用。这是"JAX 只管最重的主训,其余沿用已验证 torch 工具"
的混合管线,迁移风险最小。
⚠️ projector 若在 JAX 侧训过,目前随 npz 保存
(projector_params.npz),→ torch `projector.pt` 的转换同样在排期;
过渡期方案:JAX 冻结 projector(去掉 --train-projector),
沿用 torch stage a 的 projector 语义。

---

## 3. 必读注意事项

| 事项 | 说明 |
|---|---|
| gemma 库版本 | 已 pin 到 `09e7b48`(jax_impl 对库内部有补丁);**不要擅自升级**,升级须重跑 `poc/` 验证电池 |
| batch 大小 | 每芯 bs=1 即已打满 v6e(实测 bs2 无增益),`--per-device-bs` 保持默认即可 |
| HBM 余量 | prod 方案 + 视觉全训时峰值 ~30.7/31.25G;若 OOM,先去掉 `--train-vision` 或减小 `--val-n` |
| 首步耗时 | 编译约 2~3 分钟,属正常,勿当 hang |
| 吞吐口径 | 看日志 `marginal_micro_s`(边际耗时);epoch 时间 = marginal × 总 micro 数 |
| 改 prompt | 必须重新生成 hf_layout.json,且与端侧部署同步 |
| 完整细节 | 设计/踩坑/验证记录见 `jax_impl/FINDINGS.md` 与 `README.md` |
