# 技术路径②:K=32 输入缩减下达标(过验收线)

日期:2026-08-17(数字均为本日实测复核,n=11022)
目标:在 **K=32 输入缩减**(每帧 64→32 soft token,16 帧 1024→512)约束下,让适配器过验收线 **RT ≥ 87.91 且 SubKS ≥ 80.42**(当前 test / fit-full 口径,不管 OOF)。
这条路径解决"K=32 下的精度",与路径①(压缩体积)正交、可独立组合。

## K=32 机制
`data.py` 池化后 token 压缩:`mode=dyn`(动态选择)或 `mode=topk`(硬 top-32),K=32/64。base LLM 全程冻结,只训 LoRA + 投影 + vision。

## ★核心教训:必须"在 K=32 上训练"(匹配)
- **反例(失配崩线):** 满 token(1024)训出来的模型,推理硬套 K=32 → 严重 train/test 失配。实测 ns_repair_r64(满token训)强制 K=32(hard topk)= **RT 85.08 / SubKS 76.10**(崩)。
- **正解:** 训练期就开 K=32(dyn),让模型在 512-token 环境里学 → 匹配,不崩。

## ★过线配方:r64_rtw(K=32,零 test 接触)
`outputs/r64_rtw/`(权重在 B:outputs/r64_rtw/train_params_best.npz,484MB fp32)
配方:r64v2 基座 + **纯训练集(零 test 接触)** + `--rt-w 8`(RT 字母位加权)+ 双教师 KD(teacher_u512 + teacher2_hyb2b_u512,coef 0.5 / temp 2.0)+ 稀有类加权 + augment + 400 步,**训练&评测均 K=32 dyn**,int8。

指标(同一份 eval_int8dyn.jsonl,换校准强度):
| 校准口径 | RT(线87.91) | SubKS(线80.42) | 判定 |
|---|---|---|---|
| **每类仿射 fit-full(交付,fitfull_affine.npz)** | **88.10** | **80.49** | **双过 ✓** |
| bias-only fit-full(温和) | 87.91 | 79.95 | ✗ |
| 严格 OOF(无泄漏) | 87.84 | 79.36 | ✗ |

**验证方式**:把存档 `fitfull_affine.npz`(rt_a/rt_b/sk_a/sk_b 逐类 s·L+b)套回 logits 重算,精确复现 88.10/80.49。
**诚实标注**:80.49 靠"较激进的每类仿射"才顶过线;乐观空气 +1.13(vs 满token交付件 +0.51),比满token那版脆,换独立验收集会回落 ~79.4。在客户定的"当前 test 过就行、不管 OOF"口径下算过。

## 备选过线配方:r128 mixed_h(K=32,更稳)
`soup_tk32_r128/model_mixed_h_packed.npz`(122.2MB,混精 int8+int4)
- 每类仿射 fit-full:RT 88.12 / SubKS 80.55(bias-only 也过 80.55)
- **严格 OOF:87.97 / 80.48 —— 连无泄漏都过**(独立验收集也稳)
- 代价:比 r64 大 ~34MB,且要"逐张量混精"部署支持。

## 撞墙的方向(K=32 下 r64 过不了的口径)
- 严格 OOF / 洁净 held-out:r64 任何配方 ~79.4~80.2(8 次独立验证,512-token 容量墙)。t80(折 80% test 进训练)洁净 held-out 也只 79.41。
- ⇒ r64 在 K=32 下**只在 fit-full 仿射口径过**(80.49);要"连 OOF 都过"必须上 r128 mixed_h。

## 两条 K=32 过线选择
| 交付 | 体积 | fit-full 仿射 | 严格 OOF | 特点 |
|---|---|---|---|---|
| **r64_rtw int8** | ~88MB | 80.49 ✓ | 79.36 ✗ | 最小,余量薄,依赖激进校准 |
| **r128 mixed_h** | 122MB | 80.55 ✓ | 80.48 ✓ | 稳(含 OOF),大 34MB,需混精支持 |

## 与压缩路径的组合
路径①(压缩)× 路径②(K=32):
- r64_rtw fp32 484MB → 路径①逐通道 int8+省零 → ~88MB,K=32 fit-full 80.49。
- r128 mixed_h:soup fp32 964MB → 路径①混精打包 → 122MB,K=32 fit-full 80.55 / OOF 80.48。
