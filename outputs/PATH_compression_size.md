# 技术路径 A:体积压缩达标(≤122MB 级)

目标:LoRA 适配器从 prod 满秩(≈1GB bf16)压到 122MB 级,校准后 RT≥87.91 / SubKS≥80.42。
本路径与 K=32 输入缩减路径(PATH_k32_input.md)正交,最终交付物同时满足两者。

## 达标配方(按执行顺序)
1. **起点:同基底权重汤** —— 两个 K=32 适配成员权重平均(soup_tk32),满秩校准 ~80.87。
   汤只能同 SVD 基底/同结构;跨基底平均会摧毁权重(实测 69.56)。
2. **秩截断:act-stats 加权 SVD** —— `svd_truncate_lora.py --rank R --act-stats stats.npz`。
   激活统计加权优于普通 SVD;stats 用 `collect_act_stats.py` 在 K=32 口径收集。
3. **恢复训练:双老师合议 KD** —— 两个 u512 老师 logits 平均(`--teacher-npz` + `--teacher-npz2`),
   比单老师 +0.45 SubKS(r64 档);lr 1e-5、800 步、accum16、sw_rare 加权。
   老师 npz 必须 uniform 单一 rank(非 uniform 会在 train_sft.py:452 报错)。
4. **量化(按目标字节选档)**
   - int8 逐通道对称 + **量化恢复环**(量化→150步短KD→再量化):代价≈0。
   - int4 逐通道对称(lora 子树,`quantize_lora_int4.py`):r128 上代价仅 −0.13。
   - 逐张量混精(敏感张量 int8、其余低精):mixed_h=121.94MB,精度-字节最优前沿。
5. **校准:每类 logit 校准**(argmax 前 `logit*a+b`,共 68 标量)
   - 公平口径 = 5/2 折坐标上升折平均(`fit_calibration.py`);
   - fit-full 仿射(`fit_calibration_affine.py`)按业主判据使用,口径必须随数标注。
6. (RT 短板时)**rt-w8 定向再训**:`--rt-w 8` 单独提权 RT 字母位 400 步,
   RT +0.2(公平口径踩线),SubKS 让 ~0.35 由仿射校准收复。

## 各秩档实测天花板(K=32 口径,校准后)
| 秩档 | int8 | int4 | 结论 |
|---|---|---|---|
| r64 | 80.30(rt-w8 后 fit-full 80.49) | — | ~80.3=512token 学生容量墙,须 rt-w8+仿射 |
| r96 | 80.44(182MB) | — | 勉强过线 |
| r128 | 80.63(240MB) | **80.54(122.3MB)** | 厚余量吸收 int4 代价,最稳 |
| r128 混精 | **80.48 OOF(121.94MB)** | — | 最硬口径过线 |

## 已证无效(勿复投)
KD 加码(满分辨率老师/更高 coef)、损失侧(focal/pair-margin)、错误样本加权、
种子汤(r64 终版)、变秩 map(vrk32 78.42)、跨基底汤、GRPO/RL。

## 交付物
delivery_r128_int4 / delivery_r64_int8 / soup_tk32_r128(mixed_h)+ 各校准参数。
恢复:`gs://zx_vlm_dataset/delivery_backup_0817/`。
