# JAX 训练路线使用手册(逐阶段)

面向已在用 torch 路线的用户。JAX 路线与 torch 路线**互操作**(adapter 在
HF peft 格式层双向互换),可整体迁移,也可只把最耗时的主训搬过来。
实测吞吐(v6e,prod 超参+视觉全训,含数据预取):**9.7 样本/s@4 芯**,
8 芯外推 **100k≈1.4h / 45 万≈6.5h / 100 万≈14.3h**(torch 实测口径
100 万≈27.5h);编译 ~2 分钟(torch_xla 15~30 分钟)。

阶段顺序与 torch 版 `training/pipeline.py` 一致,逐阶段手动执行、
每阶段跑完看验收标准再决定 继续/重跑/跳过:

```
S0 环境 → S1 stage a → S2 stage b → S3 hard_mining → S4 aux_heads
→ S5 implicit_cot → S6 kto → S7 swa → S8 评测 → S9 导出交付
(S3~S7 均可选,跳过时下一阶段直接衔接最近一个已完成阶段的产物)
```

**衔接的统一规则**:每个训练阶段的产物是 `<out>/train_params.npz`
(lora + projector + aux 头,一个文件);下一阶段用
`--init-npz <上一阶段>/train_params.npz` 接上。与 torch 的
`--init-from final/` 目录语义等价。

---

## S0. 一次性准备

**方式一(推荐,团队对齐):环境镜像 + git 代码**

> **版本口径(重要,v1.8 起代码与镜像分离)**:
> | 组件 | 版本载体 | 更新动作 | 频率 |
> |---|---|---|---|
> | 代码(jax_impl/…) | git commit | `git pull` + 重启容器 | 每次修复 |
> | 环境(jax/gemma 依赖) | 镜像 tag `env-vN` | `docker pull` | 仅依赖变化,少见 |
>
> 镜像是**纯环境件**(不含代码);代码经 `-v $PWD:/workspace` 挂载进
> 容器,跑的永远是宿主机 git 仓库的版本。启动横幅 `[logtee] 代码:
> commit xxxx` 直接显示在跑的代码版本。历史 v1.x tag(含代码快照)
> 已废弃,勿再使用。

```bash
# 仓库已公开只读,免认证直接拉(注意 leeway-main 全小写;
# 报 docker.sock permission denied 时用 sudo 或把用户加进 docker 组)
docker pull europe-west4-docker.pkg.dev/leeway-main/anker/jax:env-v1
git clone https://github.com/Leeway2025/anker_vlm_training.git && cd anker_vlm_training

> **日志落盘**(v1.5+): 训练/推理/KTO 启动后自动把全部输出(含 libtpu C++
> 报错)追加写入 `<--out 目录>/train.log`;`--out` 在挂载目录下时日志即
> 持久化在宿主机,`docker rm` 不丢。`docker logs` 照常可用。启动横幅
> `[logtee] 代码: commit xxxx` 显示实际加载的代码版本与路径,
> 排"跑的哪份代码"一眼定位。

> **训练口径对齐 torch 生产(v1.7+,默认开启)**:
> ① 每 epoch 固定 seed 重洗(`--seed`);② lr 日程 warmup 300 + 线性衰
> 减到 0(`--warmup/--lr-schedule`),weight_decay=0,vision LoRA 2e-5,
> LoRA+ B 矩阵 lr×16(`--vision-lr/--loraplus-ratio`);③ val 按
> camera 整组切分、先切后复制、val 永不注入 CoT,val_n 自动对齐
> DP×BS;**评测/交付用 `train_params_best.npz`(val 最优),
> `train_params.npz` 是最后一步**。想还原旧行为: `--lr-schedule
> constant --warmup 0 --weight-decay 1e-4 --vision-lr 1e-4
> --loraplus-ratio 1`。

> **目标格式改为无空格(v1.8+,与 GT 逐字节一致)**: 训练目标从
> `"B | i | desc"` 改为 `"B|i|desc"`(torch/客户生产口径)。v1.7 及
> 之前训的 checkpoint 输出带空格 —— 评测解析两种都认,但交付口径不同,
> **正式轮请 git pull 至最新代码后训练**。另: eval_metrics 缺失预测在全部指标
> 中记错(旧版安全召回分母漏计);export/import 遇视觉塔 LoRA 无映射
> 时响亮告警(权重留在源文件,不丢失);aux 标注低置信度整条屏蔽
> (`--aux-conf-threshold`,默认 0.5,attributes 无 confidence 字段则
> 视为 1.0 不受影响)。
# TPU VM 上运行(--privileged + /dev 使容器可见 TPU;GCS 凭据走 VM metadata;
# -v $PWD:/workspace 挂载代码 —— 必挂,忘挂会直接报"找不到 jax_impl"):
docker run --rm --privileged --net=host \
  --ulimit nofile=1048576:1048576 --ulimit memlock=-1 \
  -v /dev:/dev -v $PWD:/workspace -v /path/DATA:/data -w /workspace \
  europe-west4-docker.pkg.dev/leeway-main/anker/jax:env-v1 \
  python jax_impl/train_sft.py --labels /data/labels.jsonl ...
# 代码更新: git pull && docker rm -f <容器> && 重新 docker run(运行中的
# 训练不受 git pull 影响 —— 进程用的是启动时加载的版本);
# 环境镜像发布(仅依赖变化)一律走 jax_impl/release_image.sh env-vN
```

**方式二:裸机 venv**
```bash
# ① 独立环境(python>=3.12;与 torch venv 完全隔离)
bash jax_impl/setup_jax_env.sh /path/to/venv_jax

# ② 排布模板(一次性;用现有 torch venv 生成,产物给 JAX 用)
<torch_venv>/bin/python jax_impl/poc/02a_dump_hf_layout.py --out hf_layout.json
```
- 权重自动拉取 `gs://gemma-data/checkpoints/gemma4-e2b-it`,免手动下载;
- 数据与 torch 完全同源(同一份 labels.jsonl + euno-wds 分片),零转换;
- **分片(shard-*.tar)定位优先级**: ① 命令行 `--wds-dir`(显式覆盖)
  ② labels.jsonl 各行的 `meta.wds_dir`(与 torch euno_wds 行为一致,
  路径由数据方维护,错误如实报错不静默回落) ③ labels.jsonl 所在目录兜底。
  容器提示: meta.wds_dir 是宿主机路径时,**同名路径挂载最省事**
  (`-v /真实分片路径:/真实分片路径`),jsonl 零修改;
- **验收**:`python jax_impl/poc/01_load_model.py` 打印 `Gate A: PASS`。

以下命令均用 `<venv_jax>/bin/python`,工作目录 = 仓库根;
`DATA/labels.jsonl` 与 `hf_layout.json` 两个参数每条命令都要带,下略作 `<公共参数>`:
```bash
公共参数 = --labels DATA/labels.jsonl --layout hf_layout.json
```

---

## S1. stage a —— projector 预热

- **目的**:只训视觉→文本投影(1.2M 参数),给 stage b 一个好起点。
- **前置**:S0 完成即可(从 base 权重开始)。
- **命令**:
  ```bash
  python jax_impl/train_sft.py <公共参数> \
    --stage a --steps <约1个epoch> --accum 4 --out outputs/jax_a
  ```
- **产物**:`outputs/jax_a/train_params.npz`(此阶段内只有 projector 有效)。
  同 rank 方案续接时其全零 LoRA 会被自动跳过(v1.6+,防死鞍点),日志
  出现 `全零 LoRA a 跳过 N 叶` 属预期。
- **验收**:loss 平稳下降(与 torch 版验收同口径)。
- **衔接 S2**:`--init-npz outputs/jax_a/train_params.npz`。
- **已有 torch stage a 成果?** projector.pt→JAX 的转换器暂缺(排期中)。
  由于本阶段 8 芯只需分钟级,**建议直接在 JAX 重跑**,不必衔接。

## S2. stage b —— 主训(生产超参)

- **目的**:LLM+视觉 LoRA + projector 联合训练,主要提点阶段。
- **前置**:S1 产物(或跳过 S1 从 base 开始)。
- **命令**:
  ```bash
  python jax_impl/train_sft.py <公共参数> \
    --rank-scheme prod --train-vision --train-projector \
    --init-npz outputs/jax_a/train_params.npz \
    --steps <N> --accum 4 --eval-every 50 --val-n 512 \
    --out outputs/jax_b
  ```
  `--rank-scheme prod` = 与 torch 生产同款:差异化 rank(全局层 512/
  滑动层与视觉 256)+ rsLoRA(α=2r)。
- **产物**:`outputs/jax_b/train_params.npz`;日志含 train loss、
  `[eval] val_loss`(带 *best 标记)。
- **验收**:val_loss 收敛;S8 评测 SubKS/RT 明显高于基线。
- **衔接 S3~S5**:`--init-npz outputs/jax_b/train_params.npz`。
- **已有 torch stage b 成果?** 见文末「与 torch 成果衔接」§A——
  可导入续训,但**须走 legacy 方案且不可再用 --rank-scheme prod**。

## S3. hard_mining —— 难例挖掘续训(可选)

- **前置**:S2 产物。三步:推理 → 挖掘 → 加权续训。
- **命令**:
  ```bash
  # ① 全训练集推理 —— 全芯并行(每芯一进程 × --shard 分片,断点续跑):
  # v1.6+: infer/kto 从 npz 自动判定 rank 方案(prod 产物自动带 rsLoRA
  # 缩放),并自动合并训练过的 projector;任何键未命中直接报错。
  # ⚠️ 评测 prod 产物必须用 ≥v1.6 —— 旧版会静默退化为 base 模型。
  bash jax_impl/infer_sharded.sh <venv_jax>/bin/python \
    DATA/labels.jsonl hf_layout.json outputs/preds \
    outputs/jax_b/train_params.npz
  # (单进程调试: python jax_impl/infer.py <公共参数> --shard 0/8 ...)
  # ② 挖掘(错例加权,带按类膨胀上限)
  python jax_impl/hard_mining.py --preds outputs/preds.jsonl \
    --labels DATA/labels.jsonl --out outputs/sw.json
  # ③ 加权续训(错例物理复制)
  python jax_impl/train_sft.py <公共参数> \
    --rank-scheme prod --train-vision --train-projector \
    --sample-weights outputs/sw.json \
    --init-npz outputs/jax_b/train_params.npz --out outputs/jax_hm
  ```
- **产物**:`outputs/jax_hm/train_params.npz`。
- **验收**:监控集上热点混淆对(如 h/n、k/l)减少,总指标不降。
- **衔接**:后续阶段改用 `outputs/jax_hm/train_params.npz`。

## S4. aux_heads —— 7 属性辅助头 + KS 父类头(可选,需资产 A)

- **前置**:S2/S3 产物 + Gemini 资产 A(attributes.jsonl)。
- **命令**(在 S2 命令基础上追加两个开关,换 init/out):
  ```bash
  ... --aux-file DATA/asset_A_attributes.jsonl --ks-head \
      --init-npz outputs/jax_hm/train_params.npz --out outputs/jax_aux
  ```
- **产物**:train_params.npz(aux 头参数包含在内,仅训练期使用,
  不参与导出/部署)。
- **验收**:主分类 loss 不升,RT 相对上一阶段提升。

## S5. implicit_cot —— 隐式推理链(可选,需资产 C)

- **前置**:S4(或 S2/S3)产物 + 资产 C(reasoning.jsonl)。
- **命令**:
  ```bash
  ... --cot-file DATA/asset_C_reasoning.jsonl --cot-anneal 0.5 \
      --init-npz outputs/jax_aux/train_params.npz --out outputs/jax_cot
  ```
  语义与 torch 版一致:60% 样本带推理链(think 段 loss 权重 0)、
  最后 50% 步数自动切纯生产模式(日志有 `[anneal]` 行)。
- **验收**:分类指标不降;S8 评测格式合规率必须 100%(即 think 零泄漏)。

## S6. kto —— 偏好优化(可选)

- **前置**:上游产物 + 偏好对数据(kto_data.jsonl,由推理错例构造,
  格式与 torch 的 build_kto_data 产物相同,可直接复用 torch 侧生成)。
- **命令**:
  ```bash
  python jax_impl/kto.py --kto-data outputs/kto_data.jsonl <公共参数> \
    --init-npz outputs/jax_cot/train_params.npz --steps <N> \
    --out outputs/jax_kto        # 默认全芯数据并行(--dp 0=全部)
  ```
- **验收**:日志 `|logratio|` 稳步增大(policy 在离开 ref);
  监控集分类不降(任一降 >0.5 即停,同 torch 刹车口径)。

## S7. swa —— 权重平均(可选)

```bash
python jax_impl/swa.py \
  --ckpts outputs/jax_kto/train_params.npz outputs/jax_cot/train_params.npz \
  --out outputs/jax_swa
```
- **验收**:S8 评测 ≥ 平均前的最好单点。

## S8. 评测 —— 分类指标(每个阶段跑完都建议做)

```bash
python jax_impl/infer.py <公共参数(labels 换测试集)> \
  --init-npz outputs/<某阶段>/train_params.npz --out outputs/eval_preds.jsonl
python jax_impl/eval_metrics.py --preds outputs/eval_preds.jsonl \
  --labels DATA/test_labels.jsonl --per-class
```
输出:RT / SubKS / 双对 acc、KS 父类 acc、安全关键 SubKS 召回、
格式合规率(与客户口径一致)。

## S9. 导出交付 —— 回到 HF/RKLLM 生态

> **视觉塔 LoRA 已可导出**(2026-07-20,commit cb4bbe0 起): export_hf
> `--scheme prod` 产物含 vision_tower 16 层 × 7 模块(共 714 tensors),
> RKLLM 合并链路完整;轴序经 base 权重逐位证明。同 commit 起可训集合
> 与可交付集合严格一致(PLE 门/vision entry 投影冻结)。torch→JAX
> import 方向的视觉映射暂缺(仅影响"拿 torch 视觉适配来续训",告警
> 提示,需要时再补)。

```bash
python jax_impl/export_hf.py --npz outputs/<最终阶段>/train_params.npz \
  --out outputs/final_adapter_hf --scheme prod
```
- 产物 = 标准 peft adapter(adapter_model.safetensors +
  adapter_config.json,含 rank_pattern/alpha_pattern/use_rslora),
  **与 torch 训练产物同格式**:torch 侧 run_inference、评测、RKLLM
  导出链(含 Issue #480 前缀处理)全部直接可用;
- projector 若在 JAX 侧训过,随 npz 导出为 projector_params.npz;
  → torch projector.pt 的转换器在排期,过渡期方案见下§B。

---

## 与已有 torch stage a/b 成果的衔接

### §A. torch stage b 的 adapter → JAX 续训(已数值验证)

```bash
python jax_impl/import_hf.py --adapter outputs/phase5_sft_b/final \
  --out outputs/imported_b.npz          # 真实 ckpt 对拍 PASS 的导入链
python jax_impl/train_sft.py <公共参数> \
  --init-npz outputs/imported_b.npz \
  --train-vision --train-projector --lr 1e-5 --out outputs/jax_cont
```
三条铁律:
1. **不要加 `--rank-scheme prod`**:torch 的 k/v、gate/up 各自有独立
   LoRA A,JAX 融合算子共享 A,导入用 rank 加倍拼接精确表示
   (容器 rank=1024)——与 prod 方案结构不同,**两方案不可混用**;
2. **前 50~100 步小 LR 预热**(Adam 动量不跨框架迁移);
3. 续训完导出用**默认** `export_hf.py`(不带 --scheme prod);
   rank=1024 的 adapter 是否超 RKLLM 的 LoRA 内存预算需确认。

### §B. torch stage a 的 projector.pt(转换器已提供 ✅)

```bash
# torch stage a 成果 → JAX(喂给 5b 的 --init-npz):
<torch_venv>/bin/python jax_impl/convert_projector.py \
  --torch-pt outputs/phase5_sft_a/final/projector.pt --out outputs/proj_a.npz
# 5b 命令加: --init-npz outputs/proj_a.npz(--train-projector 保持开)

# 反方向 JAX → torch(交回 torch/RKLLM 生态):
<torch_venv>/bin/python jax_impl/convert_projector.py \
  --npz outputs/jax_5b/train_params.npz --out projector.pt
```
方向经 base 权重逐位证明(w_jax = W_torchᵀ,bf16 舍入级),
每次转换内置回环自检;需在 torch venv 运行(读写 .pt)。

### §C. 混合管线(迁移风险最小的推荐形态)

```
JAX(1M≈14.3h/epoch): S1→S2(→S3 续训段)   torch(已全流程验证):
        └── S9 导出 adapter ────────▶  infer_sharded 推理 / 评测 /
                                        build_kto_data / RKLLM 部署
```
只把最重的主训搬到 JAX,其余环节沿用 torch 已验证工具,任何时刻
可整体退回 torch 路线(数据、格式、超参三层全部兼容)。

---

## 必读注意事项

| 事项 | 说明 |
|---|---|
| gemma 库版本 | 已 pin `09e7b48`(jax_impl 对库内部有补丁);**勿擅自升级**,升级须重跑 `poc/` 验证电池 |
| batch | 每芯 bs=1 已打满 v6e(实测 bs2 无增益),`--per-device-bs` 保持默认 |
| 数据预取 | **默认开启**(`--prefetch-workers 8`,并行解码+双缓冲,+30% 吞吐,loss 轨迹与同步版逐位一致);排查数据问题时用 `--prefetch-workers 0` 退回同步路径 |
| HBM | prod+视觉全训峰值 ~30.7/31.25G;OOM 时先去掉 `--train-vision` |
| 首步耗时 | 编译 2~3 分钟属正常,勿当 hang |
| 容器跑 TPU | `docker run` 必须带 `--ulimit nofile=1048576:1048576 --ulimit memlock=-1`(libtpu 需大量 fd,默认限制下报 "Too many open files" 崩溃,真机踩坑);TPU 被其它进程占用时报 vfio busy,先 `scripts/stop_train.sh` 清场 |
| 吞吐口径 | 看日志 `marginal_micro_s`;epoch ≈ marginal × 总 micro 数 |
| 改 prompt | 必须重新生成 hf_layout.json,并与端侧部署同步 |
| 细节档案 | 设计/验证/踩坑全记录见 `jax_impl/FINDINGS.md` |
