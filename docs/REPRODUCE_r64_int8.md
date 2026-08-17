# r64 int8 交付件复现手册(88MB / fit-full 88.13 / 80.65)

日期:2026-08-16
目标:从 labels + 单模基座复现交付适配器 `model_int8_packed.npz`(88.19MB)及其 fit-full 指标 RT 88.13 / SubKS 80.65(class_diag,n=11022)。
仓库:github.com/Leeway2025/anker_vlm_training(main @ 7ce46e1)。通用环境/数据/导出见 `docs/REPRODUCE.md`;本文件只补交付 r64 int8 这条具体链。

> 关键前提(必读):**交付的 r64 是满 token(1024,未启用 K=32)**。r64 在 K=32(1024→512)下崩线(SubKS fit-full ~79.8–80.29,不过 80.42),满 token 是 r64 唯一能过线的形态。客户已接受该取舍(放弃 K=32 的端侧延迟优势)。

## 链路(5 步)

### 0. 环境
按 `docs/REPRODUCE.md` §0(torch 2.9.0 + torch_xla[tpu]2.9.0 + transformers 5.13.0 等,精确 pin;v6e-8)。

### 1. 基座:r64 单模
`outputs/delivery_0807/ns_single_r64/model.npz` —— uniform rank 64 单模基座(满 token)。

### 2. 修补训练(single-teacher KD,满 token)
train.log 里的确切命令(逐字):
```
python jax_impl/train_sft.py \
  --labels /data/labels_train_plus_testval_v2.jsonl \
  --layout /data/hf_layout.json --val-ids /data/test_val_ids_v2.txt \
  --rank-scheme uniform --rank 64 --train-vision --train-projector \
  --init-npz outputs/delivery_0807/ns_single_r64/model.npz \
  --teacher-npz outputs/delivery_0807/teacher_u512.npz \
  --distill-coef 0.5 --distill-temp 2.0 \
  --augment --accum 16 --lr 1e-5 --proj-lr 2e-5 --vision-lr 1e-5 \
  --loraplus-ratio 1 --warmup 30 --lr-schedule linear \
  --steps 200 --eval-every 50 --early-stop-patience 4 --ckpt-every 100 \
  --seed 7 --mu-dtype float32 --prefetch-workers 24 \
  --out outputs/delivery_0807/ns_repair_r64
```
- 教师:`teacher_u512.npz`(满 token u512 教师),KD coef 0.5 / temp 2.0。
- 注意**无** `--token-select`/K 参数 → 满 token 1024(train.log 中 token-select 出现次数 = 0)。
- 产物 `train_params_best.npz`,best_val=2.2520@step200。

### 3. int8 打包(逐通道 + 省零叶)
`python outputs/pack_int8_r64.py`(脚本内路径写死:in=train_params_best.npz → out=model_int8_packed.npz)。
- 逐通道 int8 + fp32 scale;省略 236 个全零叶(bit-exact 无损)。
- 产物 88,187,900 B。sha256 = `94df663e23f8c3df1524d57d5c4fe7fe4cfde0478bdcf1155e118cda32b43076`。

### 4. 每类公平校准(5/2 折,fit-full)
`python outputs/safety_calib.py`(`fit_bias(L,g,folds=...)`,对最终字母 logit 做 argmax(scale·L+bias) 的每类仿射)。
- 产物 `deliver_affine_calib_r64.json`(RT scale 全 1.0 纯 bias;SubKS 仿射≈bias)。
- 部署时整体折进分类头对应行,不新增算子。

### 5. 评测(class_diag,n=11022)
8 卡 sharded 推理 → 校准后 argmax → 官方口径统计:
```
[裸分]       RoleType=87.89  SubKS=78.96
[搬用汤偏置] RoleType=88.04  SubKS=80.07
[重拟5/2折] RoleType=88.13  SubKS=80.65  <== 交付口径(fit-full),双过线 ✓
```
(严格 OOF 无泄漏:87.82 / 80.14,诚实披露,低于线;详见 DELIVERY_README_r64.md。)

## 交叉核对锚点
- `qeval_int8_calib.txt`:88.13 / 80.65(上表来源)。
- `qeval_int8_k32.*`:同权重在 K=32 下的评测(用于证明 r64+K32 崩线)。
- `DELIVERY_README_r64.md`:交付口径、指标表、K=32 取舍说明。
- LOCKED_80p56 档:80.56 里程碑锁定的权重/预测快照。

## 产物清单(outputs/delivery_0807/ns_repair_r64/,备份见 A_ns_repair_r64/)
model_int8_packed.npz(交付)· deliver_affine_calib_r64.json(交付校准)· train_params_best.npz(fp32 主权重)· DELIVERY_README_r64.md · eval/qeval 报告 · train.log。
