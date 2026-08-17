# 技术路径①:压缩达标(≤122MB int8,只算适配器)

日期:2026-08-17
目标:LoRA 适配器(base LLM 冻结,不计入)压到 **≤122MB**。
这条路径解决"体积",与路径②(K=32 达标)正交、可独立组合。

## 起点
- r64 适配器 fp32 主权重:`train_params_best.npz` **484MB**
- r128 soup fp32 主权重:`soup_tk32_r128/model.npz` **964MB**

## 三个压缩技术(叠加使用)

### 1. 省零叶(zero-omission)—— 无损
SVD/LoRA 产物里有大量**全零叶**(r64:236/507;r128:256/507)。全零叶 dequant 恒为 0,对输出零贡献 → 不落盘,部署端对缺失叶按 0 重建,**逐位等价(bit-exact)**,不改任何指标。仅此一项 r128 就省下 ~37MB 零权重。

### 2. 逐通道 int8(per-channel int8)—— 近无损
每通道独立 fp32 scale + int8 值。压缩约 5.5×,精度损失极小(int8 ≈ bf16 ±0.05,实测甚至略高)。

### 3. 混精逐张量打包(mixed_prec_pack.py)
默认 int4;按 int4 量化误差排序,把**高误差叶**在预算内逐张量提升到 int8,其余留 int4。`--budget` 控制升 int8 的叶数/总体积;`--budget 1` = 强制纯 int4(0 提升)。

## 结果(各档体积)
| 方案 | 体积 | 说明 |
|---|---|---|
| **r64 逐通道 int8 + 省零** | **88.19MB** | 压 5.5×,进预算,余 ~34MB |
| r128 逐通道 int8 | 242MB | 超预算 ✗ |
| r128 纯 int4 + 省零 | 84.9MB | 进预算(最小) |
| **r128 混精 mixed_h(int8+int4)** | **122.2MB** | 进预算,余量薄 |

## 脚本
- `outputs/pack_int8_r64.py` —— 逐通道 int8 + 省零打包(产出 model_int8_packed.npz)
- `outputs/mixed_prec_pack.py` —— 混精逐张量打包(int8/int4 择优,`--budget` 控体积)
- `outputs/delivery_0807/quantize_lora.py` —— int8/int4/int4mse/int4group 量化器(`--out-int8/--out-int4/...`)

## 部署前提(瑞芯微 NPU,交付硬前提)
1. 支持**省略零/rank-0 叶**(标准 LoRA target_modules 行为)——所有方案都要。
2. 适配器走 **int8/int4**(非强制 fp16)——否则体积账崩。
3. 若走混精 mixed_h:额外需支持**逐张量混精**(部分叶 int8、部分 int4)。

## 与验收指标的关系
压缩本身近无损(int8≈bf16;省零 bit-exact);过不过线取决于**底模精度**(见 [路径②:K=32 达标])。压缩只保证"装得下 122MB",不负责"过线"。

## sha256(交付件)
- r64 int8(满token,ns_repair_r64):`94df663e23f8c3df1524d57d5c4fe7fe4cfde0478bdcf1155e118cda32b43076`
