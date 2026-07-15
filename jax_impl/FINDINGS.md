# JAX 路线验证进展(2026-07-15)

## Gate A: PASS ✅(v6e 机器实测)

- `gm.nn.Gemma4_E2B` 实例化成功;全家族类齐:E2B/E4B/26B_A4B/31B
- checkpoint `gs://gemma-data/checkpoints/gemma4-e2b-it`,GCE 默认服务账号
  可直接读(还有 _PT 底座版);**5.12B 参数,与 HF 侧 5.1B 吻合**
- 参数树含 `vision_encoder` 与 `audio_encoder`(音频训练时剥离)
- 环境:python3.12(uv)+ jax 0.10.2 + gemma@main(v4.1.0)
  ⚠️ 依赖树带 tensorflow(seqio→tensorflow-text),盘小的机器把
  UV_CACHE_DIR/TMPDIR 指到大盘;`setup_jax_env.sh` 已处理 py>=3.12

## Gate B: 结构对齐已确认,剩模型侧占位符替换逻辑一项 🟡

HF 基准(384×384 × 16 帧 + 生产 prompt,`/dev/shm/hf_layout.json`):

```
<bos><start_of_turn>user\n
[逐帧] "MM:SS " + <start_of_image>(255999) + 64×视频占位(258884) + <end_of_image>(258882)
总长 1359;视觉 token 1024 = 64/帧 × 16(384² → 576 patch ÷ 3×3 池化 = 64)
mm_token_type_ids: 视频位置值=2(所以 sum=2048)
```

JAX 侧已确认对齐的部分:
- `Gemma4Tokenizer` 存在;**START_OF_IMAGE=255999、END_OF_IMAGE=258882
  与 HF 同 id**;占位符 258880(image)/258881(audio)连号,258884(video)
  待在完整枚举里确认
- 视觉预处理 patch16 + 3×3 池化 + max_soft_tokens=1120 → 384² 帧同样出
  64 token,**与 HF 逐帧数学一致**

Gate B 设计疑问已关闭(源码实证):
- `gemma4/_transformer.py:532`:视觉合并用 `tokens == TOKEN_PLACEHOLDER`
  (`vision/_encoder.py:32`,**哨兵值 -2**,非词表 id)做 mask 替换 →
  HF(258884)与 JAX(258880)的占位 id 差异在模型侧不存在。
  拼装配方:与 HF 完全相同的文本/哨兵 ids(255999/258882 原样),
  仅把每帧 64 个视频占位换成 -2,帧图像走 PreprocessedVisionInput。
- 剩余为机械工作:拼装 + 位置逐位 diff(并入 Gate C 前向对拍验证
  position/双向注意力语义)。

## Gate C: PASS ✅✅(纯文本 + 多模态,v6e 机器实测)

- 纯文本:同输入(显式 BOS 对齐)top-5 id 完全一致,logprob 差 <0.07
- **多模态(决定性)**:16 帧随机像素 + 生产 prompt,**top-5 id 完全一致
  且同序**,top-1 lp 差 0.0000,深尾差 <0.11 —— 像素依赖全链路
  (归一化/patchify/ViT/3×3 池化/mask 合并/位置)HF 与 JAX 数值等价

多模态拼装配方(最终版,03b_mm_parity.py 可跑通):
1. `gm.nn.Gemma4_E2B(text_only=False, config=<覆盖>)` —— 默认剥视觉塔
2. config 覆盖 `VisionEncoder(use_clipped_linears=True, output_length=64)`
   —— 默认 280 是图像语义,视频帧是 64
3. `preprocess_and_patchify(frames, max_soft_tokens=64)` —— 预算是
   "每图",默认 1120 会把 384² 放大到 1089 token
4. patches/positions 拼成 `[1, n_images*576, ·]`(模型内部按
   soft_token_counts 拆回)
5. tokens = HF input_ids,把 mm_token_type==2 的 1024 个位置换成哨兵 -2

## Gate D: API 面确认,映射表工程未做 🟡

- `gm.nn.LoRA(rank=…)` 整模包装器 + `gm.peft.LoRADense/Einsum/DenseGeneral`
  + `gm.ckpts.SkipLoRA`(加载底座跳过 LoRA)—— 注入机制齐全
- 剩余:JAX LoRA 参数树 → HF peft safetensors 名称映射 + 数值回读验证
  (04_lora_export.py 骨架已备,纯工程,无未知风险)

## 迁移决策摘要(截至 2026-07-15)

3/4 门槛实测通过,**零 NO-GO 信号**。剩余工作量(全部为已知工程):
Gate D 映射表(~1-2 天)→ 训练循环移植(加权 CE/LoRA 分层 rank/
lax.scan 累积/selective remat,~1-2 周)→ 与 torch 路线指标对拍。
预期收益:步速 1.3~1.8×(epoch 7~10h → 4~6h)+ 摆脱惰性执行 bug 类。

## 对照信息

- 客户现场:v6e-8 bs4×accum8+ckpt_on 已稳定,HBM 26.6~28.1/31.25,
  duty 75~85%
- 我方 bench:bs5×ckpt_on 可跑(无 HBM OOM);bs4/bs2 × ckpt_off 均
  HBM OOM(36.45G / 33.02G > 31.25G)→ torch 路线重算税只能靠 JAX
  的细粒度 remat 省回来,这正是 JAX 路线预期 1.3~1.8× 提速的主要来源


## 训练循环 v1 打通(2026-07-15,train_sft.py)

**结果**:假数据 64 样本、bs1、accum4、10 个优化器步 —— exit 0,
loss 8.91 → 3.79(多模态视频输入下 LoRA 真实收敛),HBM 峰值
28.72/31.25G,**稳态 micro-step ≈1.9~2.1s(torch 路线同场景 ≈5.5s),
编译仅 77s(torch_xla 需 15~30 分钟)**。

范围:LLM 层(attn+mlp)LoRA 训练(与 torch adapter 同范围);
视觉/embedder LoRA 冻结 + stop_gradient(v2 项)。

**移植攻坚记录(7 个坑,全部已修,代码内有注释)**:
1. `model.init` 在 TPU 物化 5B fp32 → eval_shape 结构化初始化;
2. one_hot 平滑 CE 物化 [L,262k] fp32 → gather 式 lse/tgt 改写 + 尾窗切片;
3. gm 库无内置逐层重算,bs1 反向 HLO 临时 66.7G → nn.remat 包 Block;
4. Block 的 bool 参数被 remat 提升成 traced array → 闭包在边界外固化;
5. fp32 LoRA 把激活链提升 fp32 → 前向 bf16(fp32 master 在优化器侧);
6. 视觉塔反向吃 ~32G(ViT 未 remat)→ v1 冻结视觉 LoRA 切断反向;
7. **gm 训练路径 remove_mm_logits 压缩视觉位 logits**(gm 生态假设
   模型自插软 token),尾部垃圾正盖住 label 窗口,表现为"末 670 位
   NaN" → 恒等旁路(我们按 HF 语义自行对齐)。

## 下一步(v2,按优先级)

1. 多芯数据并行(shard_map/pmap,4→8 卡),对拍 torch 8 卡吞吐;
2. 视觉塔 remat(vision/_modules 同款补丁)→ 放开视觉 LoRA + projector;
3. 完整导出映射表(q/kv/o/mlp 融合拆分 + 视觉塔;Gate D 已证 q_proj 链);
4. eval 循环 + 早停;真实数据对拍 torch 指标(同数据同步数 loss 曲线)。


## v2 完成(2026-07-15,commit 见 git)

| 项 | 结果 |
|---|---|
| ① shard_map 4 芯数据并行 | ✅ 10/10 步,train 8.59→3.36,HBM 28.7G/卡 |
| ② 视觉训练解锁 | ✅ VisionBlock remat + 视觉 LoRA + projector 全参(mm_input_projection),显存持平 |
| ③ 完整导出映射(export_hf.py) | ✅ q/k/v/o/gate/up/down ×35 层(490 张量)全模块 HF 对拍 PASS |
| ④ eval 循环 | ✅ val_loss 4.38→3.13,best 追踪 |

v2 新增经验:
- kv/gating 融合权重的拆分映射经**权重级对照**验证(w[0]≡gate_proj、
  w[1]≡up_proj、linear.T≡down_proj,bf16 舍入 1e-3 级);
- **自检必须 fp32 + 小扰动**:bf16 base 的精度差在 6144 大扇面模块的
  大随机扰动下被混沌放大成完全不同的 top-5,曾误判映射有错;
- checkpoint 的 projector 仅 mm_input_projection 一项(与 torch 侧
  "projector tensors=1" 观察一致);
- optax.multi_transform 分组 lr(lora 1e-4 / projector 5e-4)+
  clip_by_global_norm(1.0) 对齐 torch 超参。

## v3 残留

- 视觉塔 LoRA 的 HF 键名映射(训练已通,导出待做;torch 侧键式
  `...vision_tower.encoder.layers.N.*.linear.lora_X`);
- projector npz → HF `multi_modal_projector` 状态字典转换脚本;
- 8 卡真机吞吐对拍(需客户机或第二台 v6e);真实数据 loss 曲线对拍。
