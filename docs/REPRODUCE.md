# 客户侧复现手册(从 labels.jsonl 到交付权重)

## ⏱ 性能须知(v6e-1 真机实测,先读)

- **前几步慢是预期**: XLA 对每个新计算图编译 5~8 分钟(5.4B 模型 ×
  L2048 × 前向反向优化器)。同形状 step 编译一次后 ~2s/步,1M 数据
  训练中编译占比 <0.1%。
- **10 步后仍慢 = 在重编译**,先查: ① padding 是否固定
  (base.yaml `pad_to_fixed_length: true` 不可关)② batch 内形状是否抖动。
- **编译缓存**: 设 `XLA_PERSISTENT_CACHE_PATH` 可跨进程复用编译;
  注意部分 libtpu 版本不支持反序列化(日志出现
  "Failed to deserialize executable: UNIMPLEMENTED" 即无效,
  升级 torch_xla/libtpu 可解)。
- **逐 phase 独立进程会重复付模型加载 + 编译**,长链建议用
  `python -m training.pipeline` 编排。

## ✅ v6e-8 八卡实测结论(2026-07-09,直接照抄)

- **per_device_batch_size=4 + gradient_accumulation=8**(全局 256):
  bs8/芯 OOM 实测(34.8G/31.25G),bs4 全流程通过 —— base.yaml 已为此默认
- 启动即 `PJRT_DEVICE=TPU python training/train.py ...`,自动 8 进程;
  每核每 epoch 步数 = ceil(N_train/(8×bs)),日志核对该值即验证切分
- checkpoint/final 仅 rank0 写一份;spot 机器抢占=删除,checkpoint
  目录务必放 GCS/gcsfuse

### 排障: 启动即 KeyError: 'ACCELERATOR_TYPE'

非标准 TPU 供给(GKE 节点/自定义镜像)的 metadata `tpu-env` 键带
`TPU_` 前缀,torch_xla 默认路径按无前缀键读取 → KeyError。用 torch_xla
官方环境变量路径绕开(两个变量必须**成对**设置):

```
export TPU_SKIP_MDS_QUERY=1            # 不读 metadata
export TPU_ACCELERATOR_TYPE=v6e-8      # 补上本应从 metadata 读的值
```

只影响主机侧拓扑发现,训练数值无影响。标准 Cloud TPU VM 无需设置。

## 0. 环境

```
TPU VM(v6e-8 或同级)/ Python ≥3.10
pip install torch~=2.5 torch_xla[tpu] transformers>=4.53 peft>=0.13 \
    safetensors pyyaml opencv-python-headless decord google-genai tensorboard
python3 tests/test_core.py     # 必须全部 passed(个别用例需 torch)
```

## 1. 数据准备

```
labels.jsonl 每行(与 annotation_spec 2.2 一致):
{"video_id": "...", "video_uri": "...", "duration_sec": 14.2,
 "resolution": "1600x2200",
 "labels": {"role_type": "B", "sub_keyscene": "i", "description": "..."},
 "meta": {"camera_id": "cam_001", ...}}

⓪ euno WDS 数据(客户正式)先转换(标注/帧定位元数据齐全):
   python -m data.euno_wds --annotation <euno标注.json> \
       --wds-dir <本地或gcsfuse路径> --out DATA/labels.jsonl
   (训练是随机读,分片须本地盘/gcsfuse;gs:// 直读仅标注作业)
① 改 configs/base.yaml → data.* 路径
② 格式字节核对(整串,不止分隔符):
   从数据文件原样复制 3+ 条真实 GT 输出串存 gt_samples.txt(每行一条)
   python -m eval.check_format_alignment --gt-samples gt_samples.txt \
       --tokenizer <gemma-4 权重路径>
   全部 OK 才开训;FAIL 时按提示改 format.separator,并核对
   configs/prompt.txt 与生产 prompt 的空格/换行是否一致
③ 构造监控集: python -m eval.monitor_set --labels labels.jsonl --out monitor.jsonl
```

## 2. Gemini 增强标注(资产 A/C/D)

```
# 客户 WDS 数据(正式路径)。鉴权二选一:
#   API key 方式(客户默认): --api-key <KEY>(或 export GOOGLE_API_KEY)
#   Vertex ADC 方式:        --vertex-project <P>
python -m annotation.label_euno_wds --wds-dir <dir> \
    --annotation <euno标注.json> --out pass1.jsonl \
    --model gemini-3.1-pro --api-key <KEY> --workers 8
# (代理集视频文件用 annotation.gemini_labeler --video-root)
python -m annotation.consistency_filter --mode gt \
    --gemini pass1.jsonl --gt labels.jsonl --out-dir filtered/
python -m annotation.split_assets --gemini pass1.jsonl \
    --whitelist filtered/whitelist_ids.txt --out-dir DATA/
# ⚠️ 2026-07-10 前跑的标注: reasoning_chain 为中文(旧 prompt 未强制
#    语言)。不必重标 —— attributes/predictions/description 全部有效,
#    只需对 pass1 做纯文本翻译(实测 ~132 token/条,≈重标成本的 1~2%):
#    python -m annotation.translate_chains --in pass1.jsonl \
#        --out pass1_en.jsonl --api-key <KEY>   # Vertex 则 --vertex-project <P>
#    (断点续跑安全;之后用 pass1_en.jsonl 走 split_assets,下游零改动)
# ⚠️ split_assets 是训练消费的最后一环(资产层过滤);样本永远全量,
#    白名单不过滤训练样本(GT=Gemini+人工修正,质量高于 Gemini)
# 白名单率应 ≥85%;discarded 高频对 = Gemini 系统性盲区(报供应方
#    改标注 prompt),而非 GT 错标
```

## 3. 训练(按序执行,每步产出进下一步 init_from)

```
# torch_xla 2.x 无需外部 launcher: train.py/kto.py 内置 torch_xla.launch,
# 自动铺满本机全部 TPU 核(v6e-8 → 8 进程)。
# (旧命令 `python -m torch_xla.distributed.xla_spawn` 已随 torch_xla 2.x 消失)
S="PJRT_DEVICE=TPU python"

# Phase 5(基础 SFT)
$S training/train.py --phase configs/phase5_sft.yaml --stage a
$S training/train.py --phase configs/phase5_sft.yaml --stage b
# Hard Mining(训 3 epoch 后)
PJRT_DEVICE=TPU python -m training.run_inference --ckpt outputs/phase5_sft_b/final \
    --labels labels.jsonl --out preds_v0.jsonl
    # XLA 用 static KV cache(inference_utils 内置);断点续跑安全
python -m training.hard_mining --preds preds_v0.jsonl \
    --labels labels.jsonl --out sw.json
$S training/train.py --phase configs/phase5_sft.yaml --stage b \
    --init-from outputs/phase5_sft_b/final --sample-weights sw.json \
    --output outputs/phase5_sft_hm   # 必须换目录,勿覆盖 3b 产物

# Phase 5b(辅助头)→ Phase 5d(隐式 CoT)
$S training/train.py --phase configs/phase5b_aux.yaml
$S training/train.py --phase configs/phase5d_cot.yaml
# v1.5 推理(KTO 数据与验收共用;大规模可 --shard i/n 多机切片)
PJRT_DEVICE=TPU python -m training.run_inference \
    --ckpt outputs/phase5d_cot/final --labels labels.jsonl --out preds_v15.jsonl
# 验收: think 泄漏率必须 = 0%
python -m eval.metrics --pred preds_v15.jsonl --gt labels.jsonl | grep think_leak

# Phase 6(KTO)→ SWA
python -m training.build_kto_data --preds preds_v15.jsonl \
    --labels labels.jsonl \
    --out outputs/kto_data.jsonl
$S training/kto.py --config configs/phase6_kto.yaml
python -m training.swa --ckpts 'outputs/phase6_kto/checkpoint-*' \
    --out outputs/swa_final
```

## 4. 导出交付物

```
python -m export.split_deliverables --adapter outputs/swa_final \
    --base google/gemma-4-e2b-it \
    --projector outputs/phase5d_cot/final/projector.pt \
    --out deliverables/ --issue480   # --projector 必传,否则丢训练成果
python -m export.export_onnx --base google/gemma-4-e2b-it \
    --vision-merged deliverables/vision_merged.pt --out vision_video.onnx
# → deliverables/llm_adapter/  交给 rkllm-toolkit(base+adapter)
# → vision_video.onnx          交给 rknn-toolkit2(INT8 校准 1000 样本)
```

## 5. 每阶段验收标准

| 阶段 | 通过标准 |
|---|---|
| Phase 5 | SubKS/RT 显著高于基线;loss_cls 曲线在降(不能只有 loss_desc 降) |
| Phase 5b | RT 相对 Phase 5 +1 以上;辅助头 loss 正常下降 |
| Phase 5d | 分类不降 + think 泄漏率 = 0% |
| KTO | 监控集分类不降(离线对中间 checkpoint 评测,任一降 >0.5 → 停/换 ckpt);热区对混淆减少 |
| 导出 | 格式合规率 100%;非法组合率 <0.5%;INT8 回归 <1.5%(客户端侧) |
