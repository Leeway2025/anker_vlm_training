# 复现指南(2026-08-17)—— 三个过线交付物端到端复现

环境:TPU v6e-8,容器 tpu_train(JAX/gemma 栈),工作目录 /workspace(=仓库根)。
数据:/data/labels_train_plus_testval_v2.jsonl(训练)、/data/labels_test.jsonl(11022 评测)、
/data/hf_layout.json、/data/sw_rare_700k.json。权重资产一律可从
`gs://zx_vlm_dataset/delivery_backup_0817/` 取回。

## 0. 共同起点:满秩冠军 soup_tk32
两个 K=32 适配成员的同基底权重汤(纯训练集)。直接用备份件:
`backup: B/backup_r64_0817 上游 outputs/soup_tk32/train_params_best.npz`
(从零复现:两成员各为 prod 方案 SELECT_TOKENS_K=32 的 SFT,权重平均;详见 COORDINATION_0813.md)

## 1. r128-int4(122.3MB,公平口径 87.93/80.54)
```bash
# ① SVD 截断到 uniform r128(act-stats 可选,r128 余量厚差异小)
python3 jax_impl/svd_truncate_lora.py --in outputs/soup_tk32/train_params_best.npz \
  --rank 128 --out outputs/delivery_r128_tk32/model_r128.npz
# ② int4 仿真量化(lora=4bit逐通道对称,proj=int8)
python3 outputs/delivery_0807/quantize_lora_int4.py \
  --in outputs/delivery_r128_tk32/model_r128.npz \
  --out-int4 outputs/r128_int4b/model_int4sim.npz
# ③ 全量评测(dyn 无关此包;K=32 topk 口径)
SELECT_TOKENS_K=32 INFER_ARGS="--dump-letter-logits" \
  bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
  outputs/r128_int4b/eval_preds outputs/r128_int4b/model_int4sim.npz 8
# ④ 公平校准(重拟5/2折 行 = 官方数)
python3 outputs/delivery_0807/fit_calibration.py outputs/r128_int4b/eval_preds.jsonl \
  --gold /data/labels_test.jsonl
```

## 2. r64-int8 rt-w8 版(fit-full 88.10/80.49;公平 87.91/79.95)
```bash
# ① act-stats SVD r64 + 双师合议 KD 800步 + 量化恢复 → r64v2(或直接取备份 r64v2_fp32.npz)
bash scripts/r64v2_ens.sh        # 详见脚本;老师=soup_size/teacher_u512 + teacher2_hyb2b_u512
# ② rt-w8 定向再训 400 步(纯训练集)
bash scripts/r64_rtw_clean.sh    # 核心差异: --rt-w 8, init=r64v2, lr 5e-6
# ③ int8 + dyn 口径评测(脚本内已含)
# ④ 校准两档:
python3 outputs/delivery_0807/fit_calibration.py outputs/r64_rtw/eval_int8dyn.jsonl \
  --gold /data/labels_test.jsonl                       # 公平口径 → 87.91/79.95
# fit-full 每类仿射(交付口径;68 标量已存 outputs/r64_rtw/fitfull_affine.npz):
#   逐类坐标上升 argmax(logit*a+b),网格 a∈[0.5,1.8]步0.05 b∈[-2.5,2.5]步0.05,4遍,
#   在全量 test 上拟合(判据=用户指定的 fit-full 口径,README 已注明)
```

## 3. mixed_h(121.94MB,OOF 87.97/80.48;并行工作线产物)
资产:`A/soup_tk32_r128/model_mixed_h_packed.npz` + 同目录 DELIVERY_README.md。
要点:r128 基础上逐张量选精度(39 个敏感张量 int8,其余低精),OOF 确定性折校准。
复现细节以 soup_tk32_r128/DELIVERY_README.md 为准(含张量清单与打包脚本)。

## 4. 推理部署要点
- 选择器:dyn 全局512(每帧下限8)在 data.py `_compress_soft_tokens`,环境变量
  `TOKEN_COMPRESS_MODE=dyn SELECT_TOKENS_K=32`;NPU 侧建议三段式(编码器图→CPU .so 选择器→固定512 LLM图)。
- 校准器:argmax 前每类 `logit*a+b`(RT 5类+SK 21类),参数见各包 *_calib*.json/.npz。
- 瑞芯微三前提(省零层/逐张量混排/不强转fp16)确认后按 RKNN/RKLLM 真格式打包。

## 5. 已知坑(复现时勿踩)
1. 跨 SVD 基底的权重汤会崩(69.56)——汤只能同基底。
2. 老师 npz 必须 uniform 单一 rank(train_sft.py:452)。
3. 续训产物无 proj/ 子树时,评测前需从 init npz 并回 proj 键。
4. v2/v3 空间增强与 token 压缩管线不兼容(满帧正方形断言)。
5. 512-token 口径训的模型不要喂全 1024(训推不一致 −1.28)。
6. TPU 起任务前查 `sudo lsof /dev/vfio/0`;僵尸占卡时找 zombie 父进程 kill。
