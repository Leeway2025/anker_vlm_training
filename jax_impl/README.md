# JAX 训练路线(已投产)

Gemma 4 E2B-it 视频分类的 JAX/TPU 训练实现,全流程(S1 projector 预热 →
S2 主训 → 难例/辅助头/CoT/KTO/SWA → 评测 → HF/RKLLM 导出)已验证并在
客户 v6e-8 上实跑。吞吐约为 torch_xla 路线的 1.9×,编译 2 分钟(torch_xla
15~30 分钟)。

## 从这里开始

| 你要做什么 | 看哪里 |
|---|---|
| **跑训练/推理/评测/导出(90% 的场景)** | **[USAGE.md](USAGE.md)** —— 逐阶段命令手册,唯一权威用法 |
| 了解设计决策、验证记录、踩坑档案 | [FINDINGS.md](FINDINGS.md) |
| 早期路线论证与 Gate 验证(历史) | 本文件 git 历史 + `poc/` 目录 |

## 版本口径(2026-07-20 起,重要)

- **代码 = git commit**:`git pull` + 重启容器即完成升级;启动横幅
  `[logtee] 代码: commit xxxx` 显示在跑的版本;
- **环境 = 镜像 tag `env-vN`**(纯环境件,不含代码),仅依赖变化才更新;
  当前 `env-v1`。历史 `v1.x` tag 已废弃。

## 30 秒快速开始(TPU VM)

```bash
docker pull europe-west4-docker.pkg.dev/leeway-main/anker/jax:env-v1   # 公开只读
git clone https://github.com/Leeway2025/anker_vlm_training.git && cd anker_vlm_training
# 之后照 USAGE.md 逐阶段执行;所有命令的 docker 包装模板见 USAGE S0
```

## 三条硬规矩

1. **推理/评测永远用 `infer_sharded.sh`**(全芯并行);单进程 `infer.py`
   只用于调试——它会让 8 芯做重复计算,吞吐 = 1 芯(真机踩坑)。
2. **评测/交付用 `train_params_best.npz`**(val 最优);`train_params.npz`
   是最后一步权重。
3. gemma 库版本已 pin(`09e7b48`),勿擅自升级;升级须重跑 `poc/` 验证电池。
