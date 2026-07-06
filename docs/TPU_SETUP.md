# TPU 环境搭建与启动(v6e-8)

> 两阶段:**阶段 1 = 单卡验证环境(当前)** → 阶段 2 = 8 卡正式训练。
> 机器上已有 MaxText 的 JAX venv(~/work/maxtext/.venv)—— torch_xla
> 必须独立 venv,且**跑 torch_xla 时不能有 JAX 进程占着 TPU**。

---

## 阶段 1:单卡验证(现在做)

### 1.1 独立 venv + 安装

```bash
cd ~/work
~/.local/bin/uv venv --python 3.11 .venv-torch     # 3.10~3.12 均可
source ~/work/.venv-torch/bin/activate

# torch_xla(TPU 版;版本对齐 torch)
pip install torch~=2.5.0 'torch_xla[tpu]~=2.5.0' \
    -f https://storage.googleapis.com/libtpu-releases/index.html

pip install "transformers>=4.53" "peft>=0.13" safetensors pyyaml \
    opencv-python-headless decord google-genai tensorboard accelerate

# HF 认证(gemma-4 是 gated)
export HF_TOKEN=hf_xxx
```

### 1.2 确认 TPU 空闲(关键)

```bash
# MaxText/JAX 进程会独占 TPU,torch_xla 拿不到设备
ps aux | grep -E "maxtext|jax" | grep -v grep   # 应无输出
sudo lsof /dev/vfio/* 2>/dev/null | head        # 应无占用
```

### 1.3 跑一键烟测(单进程,自动只用 1 芯)

```bash
cd anker_vlm_training
PJRT_DEVICE=TPU python tests/smoke_tpu.py --model google/gemma-4-e2b-it
```

8 项检查依次验证:设备可见 → 模型/processor 加载 → **视频入参签名探测**
→ **PLE 命名验证(多 LoRA 红线)** → global 层检测 → LoRA 注入(rank_pattern
生效断言)→ 前向反向(首步编译/次步不重编译)→ collator 固定 padding。

任何 FAIL 都带诊断指引;全 PASS 才进入 1.4。

显存参考:E2B bf16 权重 ~10GB,v6e 单芯 32GB HBM,单卡烟测富余。

### 1.4 单卡 tiny 训练(可选,100 步)

```bash
# 用 10 条样本的 labels.jsonl 走一遍完整 train.py 流程
PJRT_DEVICE=TPU python training/train.py \
    --phase configs/phase5_sft.yaml --stage b \
    --output outputs/tiny_smoke
# 看三件事: loss 在降 / loss_cls 与 loss_desc 分离曲线都有值 /
#           final/ 里 adapter + projector.pt 都存出来了
```

---

## 阶段 2:8 卡正式训练

### 2.1 启动命令

```bash
# torch_xla 2.x 推荐 torchrun 风格:
PJRT_DEVICE=TPU torchrun --nproc_per_node=8 \
    training/train.py --phase configs/phase5_sft.yaml --stage a
# 或旧式 xla_spawn(等价):
PJRT_DEVICE=TPU python -m torch_xla.distributed.xla_spawn --num_cores 8 \
    training/train.py --phase configs/phase5_sft.yaml --stage a
```

### 2.2 batch 规划(base.yaml 当前值)

```
global batch = per_device(8) × cores(8) × grad_accum(4) = 256
v6e 单芯 32GB: bf16 权重 10GB + LoRA/Adam ~4GB + 激活(ckpt 开)~6GB
             → per_device=8 安全,可试 16(global 512)
```

### 2.3 吞吐与工期粗估(需实测校准)

```
序列 2048(视觉 1120 + 文本),2.3B + LoRA,v6e-8
预估 3~6 step/s(global 256)→ 1 epoch(95 万样本 ≈ 3.7k step)≈ 15~35 分钟
Phase 5 全程(6 epoch + hard mining 续训)≈ 3~8 小时/轮
→ 烟测后用真实 step 时间更新此表
```

### 2.4 注意事项

```
① 数据管线: 视频解码在 host CPU;8 workers × 8 进程 = 64 解码线程,
   v6e-8 host 有 180 vCPU,充足;若 dataloader 饥饿,先升 num_workers
② 磁盘: 97G 盘,checkpoint(仅 adapter+projector,~1.6GB/个)×3 保留,
   注意别与 MaxText 的 50GB ckpt 抢空间,旧 ckpt 及时清
③ XLA 编译缓存: export XLA_CACHE_DIR=~/.cache/xla 持久化,
   重启训练免重编译
④ 中断恢复: Trainer checkpoint 自带;跨 phase 断点走 final/ 约定
   (adapter + projector.pt,见 train.py 头注释)
```
