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


## v3 完成: 全流水线各阶段 JAX 化(2026-07-15)

| 阶段 | 文件 | 验证(v6e 真机,假数据短跑) |
|---|---|---|
| stage a(projector 预热) | train_sft.py --stage a | ✅ 3 步 loss 8.60→8.33,LoRA 全冻结分支正确 |
| aux 头(7 属性)+KS 头+隐式 CoT | train_sft.py --aux-file/--ks-head/--cot-file | ✅ 4 步 9.84→6.85,退火切换日志正确 |
| 推理生成 | infer.py | ✅ 2.5s/样本;固定步贪心+单张静态图,规避 generate 逐 token 重编译 |
| hard_mining | hard_mining.py(+--sample-weights 续训) | ✅ 14 错例/类膨胀上限/64→83 复制/续训 3.25→2.93 |
| KTO | kto.py | ✅ 3 步 loss 0.16~0.49,kl/logratio 正常,0.25s/micro;ref≡base 免 adapter 切换 |
| SWA | swa.py | ✅ 507 张量;异构树共有键平均 |
| **torch ckpt 导入** | import_hf.py | ✅ **真实 R1 adapter 前向对拍 PASS**(top-5 全同,lp<0.02) |

v3 关键经验:
- **torch adapter 实为差异化 rank(全局层 512/滑动层 256)+ rsLoRA +
  alpha_pattern(全局层 alpha=1024)**——import 必须逐张量取 r、逐模块
  算 alpha;alpha_pattern 正则形如 `.*\.layers\.4\..*\.q_proj`,匹配串
  必须带前导段与 self_attn/mlp 段,否则静默全漏(踩坑实录);
- 融合模块(kv/gate-up)跨框架导入用 rank 拼接精确表示,容器 R=2×max_r
  =1024;矩阵级 delta 对照(纯 numpy)是最锋利的映射验证手段;
- KTO 的 lora 树里"值为零"≠"不可导"——非训练子树必须 stop_gradient,
  否则视觉反向 62G OOM(与 SFT v1 坑 6 同根);
- KTO ref≡base(torch 的"初始 adapter 副本" B=0 数学上就是 base),
  JAX 免去 set_adapter 切换,每 micro 仅 0.25s。

**至此 torch 管线的全部阶段(a/b/hm/aux/cot/kto/swa/推理/导入导出)
均有 JAX 实现且通过机制验证;与 torch 的互操作双向打通
(export_hf 出 / import_hf 进),支持跨框架续训与混合管线。**


## v4: 每芯 bs>1 解锁 + 干净吞吐基准(2026-07-16)

- 官方源码结论: merge 侧 vmap 原生支持 batch;`_encode_vision` 写死
  B=1 → 打 batch 化补丁(data.install_batched_encode_vision),
  poc/05 等价测试 PASS(batch=2 ≡ 2×bs1,max|Δ|=9e-5)。
- **干净基准(边际计时,排除编译污染)**:
  | 配置(dp4,视觉+projector 全训) | micro 边际 | 吞吐 |
  |---|---|---|
  | bs1×4 芯 | 0.54s/4 样本 | **7.4 样本/s** |
  | bs2×4 芯 | 1.12s/8 样本 | 7.15 样本/s |
  | bs4×4 芯 | HBM 差 4.4G(35.6>31.25)| 需尾窗解码解锁,已不急 |
- **重要修正: 此前 1.9~2.1s/样本的口径是累计均值被编译污染的高估**;
  真实稳态 0.54s/样本/芯。**JAX 在 bs1 已把 v6e 打满,加大 batch 无增益**
  (与 torch 截然不同——torch bs1 饿死 MXU 才需要 bs4)。
- **epoch 推算(100k 样本)**: 4 芯 3.75h;8 芯 ≈ **1.9h**——
  相对 torch 现状(6~8h)约 3~4×,且无需再做 bs4/尾窗解码。
- 剩余吞吐杠杆重排: 选择性 remat(nothing_saveable→dots 类,+15~25%)
  与数据预取为辅;bs4/尾窗解码降级为显存优化项。

## torch 侧现场支持(同日)

- data/log_filter.py: 精确屏蔽 processor_kwargs 与 fps=24 两类刷屏
  (客户实测 50MB 日志几乎全是它),train/run_inference 入口已接;
- scripts/infer_sharded.sh: 多芯并行推理——客户现场单进程推理只有
  一张芯在算(10 万集 10+ 小时),按 TPU_VISIBLE_CHIPS 每芯一进程
  × --shard i/n 分片 → 8 芯 ≈ 8×(约 2.5h),分片可独立断点续跑。


## v5: 生产超参对齐 + eval 指标 + 健壮性(2026-07-16)

- **prod LoRA 方案落地**(prod_lora.py,`--rank-scheme prod`):
  差异化 rank(全局层 4/9/14/19/24/29/34 → 512,其余 256,视觉 256)
  + rsLoRA(α=2r → scale 32/45.25),按 scope.path 注入 gm 的
  LoRAEinsum/LoRADense Adapter。真机 4 步验证: rank 落位正确、
  loss 下降、HBM 30.72/31.25G(**很满** —— 长跑若 OOM,首选把
  lora 的 adam 状态降 bf16 或收 r_vision)。
- **prod 导出链对拍 PASS**: export --scheme prod 产出与 torch 生产
  adapter 同款配置(rank_pattern/alpha_pattern/use_rslora);
  线性区扰动下 HF/JAX top-5 完全同序。
- **对拍方法论最终版(三次踩坑换来的)**:
  1) 行为对拍必须在线性区: 扰动幅度按前向缩放反除
     (ΔW ≪ 权重,否则近平局混沌重排 top-5 出假 NO-GO);
  2) 金标准是跨侧矩阵级对照(JAX 张量按 JAX 公式 vs 导出张量按
     peft 公式,纯 numpy);已内嵌 selftest;
  3) 观察到"两侧不对称漂移"才是真 bug 信号(一侧≈base 一侧大偏)。
- eval_metrics.py: RT/SubKS/双对/KS 父类 acc + 安全关键召回 +
  格式合规率(客户口径,零 torch 依赖),假 preds 验证通过。
- 健壮性: gemma pin @09e7b48(setup_jax_env.sh,升级须重跑验证电池);
  测试资产已备份 gs://leeway-main-ml-tmp/jax_assets_20260716.tgz
  (注: 4 卡机 VM scope 只读,备份走本地跳板上传)。
- 约定重申: torch ckpt 续训走 import_hf 的 legacy 容器方案;
  JAX 原生训练/导出走 prod 方案;二者不得混用。


## P0: 数据预取 + 并行解码(2026-07-16,1M 规模硬需求)

- prefetch.py: 线程池并行解码(PIL/numpy 在 C 层释放 GIL)+ 深度 2
  双缓冲 + 异常透传(防无声卡死)+ 退火 flush(边界零滞后);
  train_sft `--prefetch-workers`(0=同步调试路径)。
- **验收(A/B 双绿)**: ①loss 轨迹与同步版逐步完全一致(每步小数点
  后 4 位全同)→ 保序保内容零语义变化;②边际 0.54→0.41s/micro,
  吞吐 7.4→**9.7 样本/s**(4 芯,prod+视觉全训),**+30%**。
- 途中抓到的新坑(异常透传自证价值): gemma 预处理的 kauldron
  `@typechecked` 用全局非线程安全 scope 栈,并发必炸
  (`scope.py assert s == self`)→ 训练进程关闭 ktyping
  (_disable_ktyping,开发期辅助,无语义影响且省调用开销)。
- **epoch 账更新(8 芯外推)**: 45 万样本 ≈6.5h,**100 万 ≈14.3h**
  (torch 实测口径 100 万 ≈27.5h;注: torch 客户实测 25.4s/it 好于
  此前估计,两路线差距修正为 ~1.9×)。
- P1(尾窗解码+选择性 remat)按用户指示押后。


## P0b + Docker(2026-07-16)

- infer.py 增 --shard i/n + 断点续跑;jax_impl/infer_sharded.sh 全芯并行
  (自动检测芯数)。**并发单芯进程隔离配方(实测)**: 仅
  TPU_VISIBLE_CHIPS 会撞 libtpu 锁("TPU already in use"),须加
  TPU_PROCESS_BOUNDS=1,1,1 + TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 +
  每进程独立 TPU_PROCESS_ADDRESSES/PORT + CLOUD_TPU_TASK_ID=0
  —— libtpu 层配方,torch 版 scripts/infer_sharded.sh 已同步加上。
  验证: 2 进程并发 merged=4 ✓ 续跑跳过 ✓。
- kto.py 多芯数据并行(--dp,默认全部): shard_map + batch 维滚动
  错配对 + 梯度 pmean;验证 4 芯 3/3 步 ✓。
  **坑: 闭包捕获的 gm 加载参数带全局分片标注('devices' Auto mesh),
  与 shard_map Manual mesh 冲突 → 参数必须显式作为 shard_map 入参**
  (train_sft 一直如此故无恙)。
- Docker: jax_impl/Dockerfile(python3.12-slim,全 pin jax 0.10.2 +
  gemma@09e7b48,构建期自检导入),镜像 5.19GB,推送
  europe-west4-docker.pkg.dev/leeway-main/anker/jax:{v1,<git sha>}
  (AR 同区,团队授 artifactregistry.reader 即可 pull)。


## P0c: projector 双向转换器(2026-07-16)

- convert_projector.py: torch projector.pt ↔ JAX npz。方向用 base 权重
  逐位证明: JAX mm_input_projection/w (768,1536) = HF
  embed_vision.embedding_projection.weight (1536,768) 的转置
  (maxdiff 0.0014 = bf16 舍入)。
- 验证: R1 真实产物 torch→jax→torch 回环 maxdiff=0.0、键名一致;
  prod 训练产物 J2T 正确;纯 LoRA npz 正确拒绝。
- 踩坑: 泛匹配 "mm_input_projection" 会误抓 embedder 的 LoRA 键
  (lora/embedder/.../lora/a)→ 必须锁 proj/ 子树 + 形状断言。
- 至此 torch stage a/b 成果均可完整接入 JAX(§A adapter + §B projector),
  跨框架互操作缺口仅剩视觉塔 LoRA 的导出映射(JAX 训视觉→回 torch)。
