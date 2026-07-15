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

剩余一项(Gate B 收尾):
- JAX 库预处理是图像语义(占位符 258880),HF 视频用 258884。输入序列
  我们可以手工拼装(ids 全对齐没问题),关键是 **gemma4/_transformer.py
  里视觉 embedding 替换逻辑按哪个 id 找位置**——若只认 258880,要么改一
  行替换 id,要么确认模型本身支持视频占位。读源码 + 单帧前向即可关闭。

## Gate C/D: 未开始(等 B 关闭)

## 对照信息

- 客户现场:v6e-8 bs4×accum8+ckpt_on 已稳定,HBM 26.6~28.1/31.25,
  duty 75~85%
- 我方 bench:bs5×ckpt_on 可跑(无 HBM OOM);bs4/bs2 × ckpt_off 均
  HBM OOM(36.45G / 33.02G > 31.25G)→ torch 路线重算税只能靠 JAX
  的细粒度 remat 省回来,这正是 JAX 路线预期 1.3~1.8× 提速的主要来源
