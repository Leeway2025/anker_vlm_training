# 客户侧复现手册(从 labels.jsonl 到交付权重)

## 0. 环境

```
TPU VM(v6e-8 或同级)/ Python ≥3.10
pip install torch~=2.5 torch_xla[tpu] transformers>=4.53 peft>=0.13 \
    safetensors pyyaml opencv-python-headless decord google-genai tensorboard
python3 tests/test_core.py     # 必须 13/13 passed
```

## 1. 数据准备

```
labels.jsonl 每行(与 annotation_spec 2.2 一致):
{"video_id": "...", "video_uri": "...", "duration_sec": 14.2,
 "resolution": "1600x2200",
 "labels": {"role_type": "B", "sub_keyscene": "i", "description": "..."},
 "meta": {"camera_id": "cam_001", ...}}

① 改 configs/base.yaml → data.* 路径
② 字节核对: 抽 3 条真实 GT 输出串,确认与 configs/prompt.txt 的
   " | " 分隔符逐字节一致(不一致改 format.separator)
③ 构造监控集: python -m eval.monitor_set --labels labels.jsonl --out monitor.jsonl
```

## 2. Gemini 增强标注(资产 A/C/D)

```
python -m annotation.gemini_labeler --labels labels.jsonl \
    --video-root videos/ --out pass1.jsonl --temperature 0.1
python -m annotation.consistency_filter --mode gt \
    --gemini pass1.jsonl --gt labels.jsonl --out-dir filtered/
# filtered/whitelist_ids.txt → configs/base.yaml data.whitelist_file
# 白名单率应 ≥80%;discarded 抽 500 条人工看(可能是 GT 错标)
```

## 3. 训练(按序执行,每步产出进下一步 init_from)

```
S=python -m torch_xla.distributed.xla_spawn --num_cores 8

# Phase 5(基础 SFT)
$S training/train.py --phase configs/phase5_sft.yaml --stage a
$S training/train.py --phase configs/phase5_sft.yaml --stage b
# Hard Mining(训 3 epoch 后)
python -m training.inference_utils_run --ckpt outputs/phase5_sft/final \
    --labels labels.jsonl --out preds_v0.jsonl        # 见 inference_utils
python -m training.hard_mining --preds preds_v0.jsonl \
    --labels labels.jsonl --out sw.json
$S training/train.py --phase configs/phase5_sft.yaml --stage b \
    --sample-weights sw.json

# Phase 5b(辅助头)→ Phase 5d(隐式 CoT)
$S training/train.py --phase configs/phase5b_aux.yaml
$S training/train.py --phase configs/phase5d_cot.yaml
# 验收: think 泄漏率必须 = 0%
python -m eval.metrics --pred preds_v15.jsonl --gt labels.jsonl | grep think_leak

# Phase 6(KTO)→ SWA
python -m training.build_kto_data --preds preds_v15.jsonl \
    --labels labels.jsonl --whitelist filtered/whitelist_ids.txt \
    --out outputs/kto_data.jsonl
$S training/kto.py --config configs/phase6_kto.yaml
python -m training.swa --ckpts 'outputs/phase6_kto/checkpoint-*' \
    --out outputs/swa_final
```

## 4. 导出交付物

```
python -m export.split_deliverables --adapter outputs/swa_final \
    --base google/gemma-4-e2b-it --out deliverables/ --issue480
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
| KTO | 监控集分类不降(任一降 >0.5 自动停);热区对混淆减少 |
| 导出 | 格式合规率 100%;非法组合率 <0.5%;INT8 回归 <1.5%(客户端侧) |
