# 项目交接文档 · HANDOFF

> **给接手的 Claude Code:** 读完本文件即可继续本项目。本文件是自洽的——它刻意把仓库代码之外的信息(项目本质、客户约束、方法论定律、当前进度、历史踩坑、机器拓扑、认证配置)全部收纳进来,因为这些原先存在开发机的 `~/.claude` memory 里,不随 git 走。**先通读第 1~6 节建立全局,再看第 9 节"当前状态与下一步"接手干活。**
>
> 最后更新:2026-08-03。口径日期均为绝对日期。

---

## 1. 项目本质(authoritative)

**任务**:Anker eufy 家庭安防(可视门铃 / 庭院摄像头)视频分类。**不是**小区/写字楼监控。权威定义源:`gemma-anker/Anker场景数据定义0706.pdf`。

**模型**:`google/gemma-4-e2b-it`(报告里写 "Gemma-4n E2B-it";E="Effective",PLE 参数下放;35 层 dense,混合注意力 512 滑窗+全局,262K 词表)。**基座锁定 Gemma 家族**——用户是 Google 员工,组织要求不得换 Qwen/Llama。

**每次推理输出**:`{RoleType 字母} | {Sub-keyscene 字母} | {~25 词英文 description}`
- **RoleType(RT)5 类**:A=Family Member, B=Staff, C=Suspicious Person, D=Unspecified, E=Non-Human(各有二级 sub_role_type)
- **Sub-keyscene(SubKS)21 类**:a~u,归属 6 个 Key_scene(Normal Activity 13 / Property Damage 2 / Life-Threatening 3 / Loitering 1 / Vehicle Anomaly 1 / Unauthorized Entry 1)
- **KeyScene(KS)**:6 大类,由 SubKS 上卷

**评测**:分类 P/R/ACC/混淆矩阵 + per-class Recall@preset-Precision;description 用 LLM-as-judge 对 GT 打分(0-100)。

**客户确认的关键约束(2026-07-06):**
1. description GT 训练数据里就有,质量 OK → 直接 SFT 学 description。
2. **description 没有任何指标目标**(用户两次强调升级确认):分类指标是唯一目标,description 允许退化(EunoVLM 的 description 也从未被测)。已应用:分类 token loss ×4、KTO 只用分类偏好、全程不挂 judge 监控、只查输出格式完整性。
3. 无 camera_type 字段;但 k/l(Leaving/Approaching Porch)定义随视角不同 → 用 Gemini 打 view_type 辅助属性缓解。分辨率↔设备映射:16:9(1920×1080/3840×2160)=户外/PTZ,1600×2200 竖屏=门铃 → 客户数据 view_type 由分辨率规则免费得到。
4. 只看统一指标,无 per-class recall 目标 → 自然分布训练成立。
5. **PDF 里的字母码 prompt 就是固定生产 prompt**,训练数据必须用这个精确格式,不能缩短。
6. **生产预处理(2026-07-06 全部确认)**:视频 5~40s;采样 = **整段均匀 16 帧**(不是固定 FPS);resize = **直接拉伸到 384×384**(不保宽高比,竖屏被横向拉宽 ~38%,基线就这么训的,照抄;RandomCrop 必须在 resize 之前)。

**任务级坑:**
- `h`(Package Brought Home)vs `n`(Package Taken Away)视觉上完全相同,只能靠人物身份区分 → RT 与 SubKS 强耦合,必须联合训练;专门盯 h/n 混淆。
- `C`(Suspicious Person)在 spec 里没有正向定义 → 标签噪声源;锚定 SubKS∈{s,t,u,n,q} 相关性。
- 大小写混淆:"C"(suspicious)vs "c"(kid playing)只差大小写 → 需输出校验。
- 基线训练配置:70 tokens(384×384)×16 帧,ViT full + LLM LoRA r=64/α=128,LR 1e-6/3e-5,3 epoch。

---

## 2. 硬约束(违反即废掉整条部署链)

这些约束**收窄了可用优化手段**,推荐任何方案前先过这一节:

1. **多 LoRA 热切换部署**:客户真实拓扑 = base Gemma 4 E2B + 多个可切换 LoRA(LoRA A=视频分类,LoRA B=function calling,还有更多)在 Qualcomm NPU 上热切。
   - **base 参数(Vision Encoder / LLM 主干 / PLE)必须冻结**——base 一动,所有其他 LoRA 的 delta 全部失效。
   - **不训 PLE**(共享 base 一部分,训了毁掉 function calling)。
   - **不做全参 ViT / projector 微调**(共享)。
   - 视频任务 **只能改 LoRA adapter 权重**;ViT/projector 若要调也只能挂 LoRA(随视频 LoRA 一起加载)。
   - EunoVLM 未测 function calling,极可能灾难性遗忘了工具调用能力("专精税"),客户付不起。
2. **Qualcomm NPU 图不变**:QNN 编译器对官方 Gemma 4 拓扑做静态图分析,任何未知子图(如插入 Resampler)→ 编译失败或 CPU 回退。
   - **R1 类改动(仅配置)对部署免费**:FPS、帧数、visual token 预算(支持值内)、LoRA rank、训练数据、prompt。
   - **R2 类改动(改图拓扑)极贵**:任何新模块破坏 Super Node 融合,只有 R1 耗尽且收益足够大才值得。
   - **重参数化逃生舱**:训练时复杂多分支模块,推理时数学折叠回标准算子(RepVGG/LoRA 式)。可折叠的:多并行 Linear 相加、Linear→Linear 无非线性链、LoRA merge、固定 affine 的 BN/LN、数据无关 pooling、时间维 1D conv。**不可折叠**:softmax 注意力、内容相关权重(QK^T)、线性间的 GELU/ReLU/SiLU、token 数变化。
3. **低延迟,禁止推理时深思**:边缘产品延迟敏感。Thinking Mode、多样本投票、long-CoT prompt、test-time search 一律**不可接受**(用户明确拒 Thinking Mode "太慢")。
   - ✅ 允许(只花云端训练算力):训练数据、训练配方、LoRA rank、多阶段预训练、蒸馏(学生保持小)、训练期辅助头、**隐式/内化 CoT 蒸馏**(训模型"默想")、重参数化。
   - ❌ 不允许(增加设备端 token):推理时 Thinking Mode、多样本投票、long-CoT、RAG、test-time search。
   - **重要推论**:凡本项目做 CoT,都是"隐式 CoT"——训练时喂推理链、think 段权重置零,推理时**不生成**任何思考 token。
4. **视觉 token 预算保持低(140~280),不要为"细粒度"调高**:高层语义分类任务,时序密度 >> 空间清晰度。EunoVLM 9 tokens/frame 就到 95.34% KS。高 token 会被背景空间噪声淹没 LoRA 信号,且伤害滑窗时序建模。把预算投给更多帧/更高 FPS。
5. **JAX-only(2026-07-21 用户拍板)**:训练/推理/评测只走 JAX(`jax_impl/`);torch 训练代码归档(仅 RKLLM 端侧导出链继续用 torch 工具)。**超参以 JAX 端实测独立标定,不再对齐 torch**。

**分工**:我方(Google 侧)只做**云端训练**;客户(Anker 侧)负责全部端侧(RKLLM 工具链、RK1828 多 LoRA runtime、.rknn/.rkllm 编译、INT8 量化、设备端延迟验证)。我方交付边界 = HF 格式权重(base 不动 + LoRA adapter + 合并的 VE + Projector)、ONNX 导出、校准样本清单、workaround 脚本、验收标准文档。**不做**编译产物/设备端运行。

**交付模式**:客户 1M 标注数据**不出其环境**。我方在代理公开数据集上验证训练管线(我方指标只是 sanity check,不与客户基线可比),交付**训练代码仓库**给客户在其 1M 数据上重训,远程支持。→ 不要规划任何需要直接访问客户数据的步骤。

---

## 3. 评测口径(动分数前必读)

- **测试集冻结 11,022 条**(`/data/labels_test.jsonl`)。**禁用 test 标签算先验**——唯一合法手段是两折交叉(把 test 分两折,一折标定先验、另一折评,交换,不触碰真标签)。
- **选模标尺钉死**:固定 val 卷(`/data/val_ids_v2.txt`,1536 条 test-mix 固定卷),消选模噪声。1M 阶段用 `val_ids_1m.txt`。
- **只对标 EunoVLM v3.0.74 @100k**:KS 90.02 / **SubKS 73.83** / RT 85.18。SubKS 是主指标。
- **训练池∩测试集重叠 ≥705 条(6.4%)**:已结案(2026-08-01 用户拍板)——**保留不剔**,EunoVLM 同池同法,对比口径对等,双方分数含同等记忆成分。1M 池同样不剔重叠(`profile_pool.sh` 仅信息披露)。**此议题不再重提。**

---

## 4. 战绩线(本轮 100k 阶段)

| 版本 | SubKS | RT | KS | 说明 |
|---|---|---|---|---|
| 上周交付(v4+先验) | 73.21 | 83.27 | 92.06 | — |
| seed 摇号 + 固定 val 卷 | 73.52 | 83.37 | 92.66 | 单发最大涨幅;同配方换种子(--seed 1) |
| **单模交付口径** | **73.77** | **83.65** | **92.74** | + 手术先验 + RT 阈值(两折交叉);距基线 0.06 ≈ 7 条视频 |
| **集成口径** | **74.44** | **84.20** | **92.74** | **SubKS 首超 EunoVLM +0.61,KS 领先 +2.72** |

**里程碑**:2026-07-31,双模 logits 集成首次跨线。历史:2026-07-22 JAX 冷配方追平客户自训 E2B 基线(对拍闭环,同精度 ~1.9× 吞吐,JAX 主路线成立);v4(去重+augment+早停)72.38 超客户 E2B 基线 +2.79。

---

## 5. 方法论定律(可复用资产,血泪换来)

1. **权重汤 vs logits 集成**:"**合体要求同盆地,合议只要求错误去相关**"。不同种子(不同损失盆地)的权重平均全灭(seed-1×replica 三档配比 73.10/73.49/73.37 均 < seed-1 的 73.52;加权汤 0.6/0.75/0.85 均 <73.52);同两个模型的字母 logits 集成 +0.67 大胜(α 两折标定选 0.7/0.8)。→ 想合模型,优先 logits 集成,别做权重汤,除非同盆地(同种子续训分支)。
2. **A↔D 跷跷板恒定律**:RT 混淆矩阵中 A→D + D→A 两格之和跨版本恒定(~950±20:v2=983, v4replica=944, seed-1=967)。→ **判决阈值类手段数学性出局**,只有身份特征学习能压缩 RT。
3. **CoT 泄题死因(S5 −2.69 的根因)**:隐式 CoT 训练链内显式含答案字母(RoleType X/Sub-keyscene y),与 GT 一致率 **100.00%**(69707/69707);60% 样本上文写着答案,字母预测退化为抄写,视觉→字母通路半程无梯度。think 段权重置零防了"学写链",没防"上下文漏答案"。**修复 = 去字母化**(`scripts/strip_cot_letters.sh`),重训排期未跑。
4. **离线解码器家族已系统性关账**:全类偏移 / RT×SK 联合解码 / τ 强度网格 / RT 全维阈值,四个自由度两折搜完,总收益 +0.11 后**饱和**。→ 别再投离线解码,除非有新自由度。
5. **懒标 D 浓度实测**:训练集 GT=D 的推理链中 **25.4%**(6339/24994)含住户级证据("自信铲雪""熟练用耕地机"仍判 D)= 测试侧 RT 决算 A 桶 D=491 同源。→ 1M 标注管线需稀有类双人标注 + 三方对质抽检。
6. **RT loss 结构性饥饿**:RT 是输出串第一个 token,占总 loss 千分之几,自回归中看不到 SK。缓解:`--rt-w`(首2字符×8)、`--idw`(desc 身份词×3,正则见 `jax_impl/data.py` 的 `_ID_WORDS`)。
7. **过拟合门禁铁律(v7 沉淀)**:任何训练配方变更先过"合成可分数据过拟合 100%"门禁再上真数据;烟测输出异常必须跑 base 对照组;**loss 正常下降不证明分类学会**(desc 会撑住曲线)。
8. **标注文件血统指纹(红线)**:`consistency_filter --mode gt` 的 rate:**≈0.373=盲判版可用**;**≈0.996=标签在 prompt 里的泄漏版,禁用**(属性从标签倒推)。任何文件切资产前先跑指纹。

---

## 6. 关键文件地图

**仓库**:`https://github.com/Leeway2025/anker_vlm_training.git`(本文件在 `docs/HANDOFF.md`)。

### jax_impl/(JAX 主路线,唯一训练/推理/评测入口)
- `train_sft.py` — SFT 主训练。`--init-npz` 续训覆盖同名叶;`--idw`/`--rt-w` 传 loss 加权;`--seed`。
- `data.py` — `SftDataset`,`rt_weight`/`id_weight` 权重回填(首2字符=RT位、cls前缀、desc身份词 `_ID_WORDS`)。
- `infer.py` / `infer_sharded.sh` — 固定步贪心推理(2.5s/样本),8 分片。
- `eval_metrics.py` — 评测(RoleType_acc/SubKeyScene_acc/KeyScene_acc + 混淆矩阵)。
- `kto.py` — KTO 续训(ref≡base,只用分类偏好)。`grpo.py` — GRPO(封存,reward 太稀疏失败)。
- `apply_surgical_prior.py` / `dump_priors.py` — 手术先验(两折交叉)。
- `dedup_labels.py`(池内去重,**从不剔 test**)、`build_v5_data.py`、`export_hf.py`(490 张量映射,HF 对拍 PASS)、`import_hf.py`。
- `FINDINGS.md` / `USAGE.md` — JAX 实现踩坑史与用法(**改 jax_impl 代码必更新 USAGE + 走镜像同步铁律,见第 8 节**)。

### scripts/(客户侧一键执行件;全部 `git pull + bash 一行`,开头 `unset WDS_DIR`)
- **集成**:`ensemble2.sh`(★ 产出 74.44:replica 裸 logits + seed-1 α两折加权平均 + m 先验 τ0.7 + RT 阈值 → `preds_ens.jsonl`)。
- **续训候选**:`kto_run.sh`(KTO 一条龙:挖偏好对→400步→评测+先验,`lora_params_best.npz` 仅超基线才落盘)、`night_seed2.sh`(seed-2 摇号+双强汤)、`night_rtw.sh`(④RT位加权)、`stage2_rtfix.sh`(二阶段续训模板 BASE=seed-1 --lr 2e-6 --steps 400 --rt-w 8 --idw 3)。
- **CoT**:`night_s5.sh`(隐式 CoT 训练)、`strip_cot_letters.sh`(去字母化手术刀)、`check_cot_asset.sh`/`check_cot_labels.sh`/`check_cot_rt_bias.sh`(资产体检三件套,硬伤退出码非0挡夜链)。
- **标注**:`rationalize_rt.sh`(RT 强化重标,只标 GT∈{A,D} 约5.15万行)。
- **离线解码(已饱和,存档)**:`rt_boost.sh`/`rt_joint_decode.sh`/`tau_grid.sh`/`subks_combo.sh`/`soup_full.sh`/`soup_weighted.sh`。
- **归因**:`rt_attribute.sh`、`day_run.sh`(白天生产线状态机,含幂等闸)。
- **1M 数据工程**:`profile_pool.sh`(池体检)、`pool_sample_blind.sh`/`pool_blind_report.sh`(盲判)、`build_pool_v1.sh`(出池:去重+坏行/空desc/中文desc剔除,**不配平不剔重叠**,+`val_ids_1m.txt`)、`check_wds_pool.sh`、`smoke_1m.sh`、`train_1m.sh`(8000步/eval250/patience4)。

### annotation/
- `rationalize_cot.py` — CoT/RT 重标注生成器(见第 7 节坑:模型代号 & 认证)。

### docs/
- `PLAN_1M.md` — 1M 五阶段作战手册(P0盘点/P1体检/P2盲判/P3策展/P4工程/P5冒烟/全量)。
- `WEEK_PLAN_20260803.md` — 本周逐日明细(三资源线并行 TPU/CPU+网络/GPU)。
- `REPRODUCE.md` / `TPU_SETUP.md` / `WALKTHROUGH.md` / `SCHEDULE.md` — 复现/环境/走查。

### 仓库外(不进 git)
- `~/gemma-anker/logs/` — 客户回传日志/截图/历史评测(`jax5b_*_eval/` 含 report.json + cm_role.png)。
- `~/gemma-anker/PROGRESS_SUMMARY.md` / 各 REPORT_*.pdf — 汇报文档(用户明确不进 git)。

---

## 7. 历史踩坑清单(重复即浪费)

1. **WDS_DIR 环境污染(集成两连败真凶)**:rationalize 里 `export WDS_DIR=训练集目录` 被 TPU 推理脚本继承,测试视频去训练 tar 找 → KeyError。**修复:所有 TPU 链脚本开头 `unset WDS_DIR`**(已做,新脚本务必照做)。
2. **模型代号 404**:客户项目 `gemini-3.1-pro` 已下线;**用 `gemini-3.1-pro-preview` 或 `gemini-3.5-flash`**。`GOOGLE_CLOUD_LOCATION=europe-west4` 会把 Vertex 请求路由到无模型区 → **用 `global`**。
3. **logits 文件覆盖事故**:重放旧 dump 命令覆盖了 `outputs/optin/preds.jsonl`(seed-1 logits),下游读出复刻水位。**修复:脚本加底座门禁(裸 SubKS<0.730 拒跑)+ 链尾自动补 dump**。
4. **TPU 芯片释放延迟**:kill 进程后需 `sleep >10s`(libtpu 锁)再启新进程,否则分片抢跑崩溃。
5. **v6e-8 host OOM(2026-07-14 结案)**:transformers≥4.46 `get_batch_samples` 一次性取完累积窗口,32 次 fwd+bwd 展开成单张巨型 HLO 图 → host 内存爆。修复见 memory `tpu-oom-root-cause`(现已 JAX-only,主要作历史参考)。
6. **PDF 渲染**:markdown extensions 必须含 `'fenced_code'`(否则代码块不解析、被顶成大标题)。流水线:python-markdown → weasyprint(Noto Sans CJK SC)→ pdftoppm。
7. **SSH 远程 pkill -f 自杀**:模式含命令自身字符串会匹配到 bash -c 自身;用 `[x]` 括号技巧或锚定 `^`,优先用具体 PID。

---

## 8. 镜像-代码-文档同步铁律(2026-07-20 起)

代码与镜像**分离**:
- 镜像 = 纯环境件(jax 0.10.2 + gemma pin 依赖,不含代码),tag `env-vN`(现行 env-v1);**仅当依赖变化**才 `sudo bash jax_impl/release_image.sh env-vN`(`sudo` 勿带 `-E`,会读错凭据报 Unauthenticated)。
- 代码 = git commit,运行时 `-v $PWD:/workspace -w /workspace` 挂载;**改 jax_impl 代码的发布 = git push,客户 git pull + 重启容器即生效**。
- 改 `jax_impl` 必更新 `jax_impl/USAGE.md`;排障问两个号:git commit + env tag。
- 历史 v1~v1.8 tag(含代码快照)已废弃,勿引用。

---

## 9. 当前状态与下一步(接手从这里开始)

### 最后已知状态(2026-08-03 周一早)
- 100k 阶段方法路径已验证可行(集成 74.44 超基线)。战略转向:**打完 100k 收尾 → 全面转 1M 数据工程**(用户:"弄了这么大的 lora 也没有啥意思")。
- **周末三条夜链(在客户 TPU v6e-8 上跑)只跑到第一步**:仅 `outputs/kto_rt/train_preds.infer.log` 存在(KTO 挖偏好对所需的全训练集推理),KTO 训练本身 / seed-2 全链 / rationalize_rt 标注**均无产物**。需先诊断 KTO 推理是静默跑完没触发下一步、还是中途崩(查 KeyError/OOM,见第 7 节 WDS_DIR)。

### 机器拓扑
- **客户 TPU v6e-8**:所有训练/推理/集成/标注在这跑。执行方式 = 发脚本进 `scripts/`,转发客户 `git pull + bash 一行`,产物落挂载盘,评测报告回传进 `logs/`。
- **我方 GCP 工作站**(原开发机 `admin@leeway.altostrat.com`):写代码、出文档、分析。
- **JAX 验证机**:随用随建(原验证机已回收),配方 + GCS 资产见 memory `jax-validation-env`(`gs://leeway-main-ml-tmp/jax_assets_20260716.tgz`)。
- **GPU 线**:1M 阶段用于切帧/预处理并行。

### 待办(优先级序)
1. **诊断并接上 KTO 续训**(单模跨线候选一):看 `outputs/kto_rt/train_preds.infer.log` 尾部与报错,train_preds.jsonl 生成没;OK 则从挖对续跑 `kto_run.sh`。
2. **seed-2 全链**(候选二):`night_seed2.sh`,seed-2 裸分 + 与 seed-1 双强汤,新原料并入集成。
3. **rationalize_rt 标注**:A/D 两类 5.15 万行,产出 `asset_C_rtfocus.jsonl`。
4. **去字母化 CoT 重训**:`strip_cot_letters.sh` 后 `night_s5.sh` 重训(资产合并后)。
5. **stage2_rtfix**:RT 位 + 身份词二阶段续训(BASE=seed-1)。
6. **1M 数据工程 P0-P5**:见 `docs/PLAN_1M.md`。这是通往 76+ 的最大杠杆(数据量 +2~4)。
7. **我欠的配套**:train_sft 断点续训(checkpoint+optimizer 落盘/resume)、三模集成脚本、GPU 切帧参数建议书。

### 过程纪律(固化)
- 每轮**单变量**、**全表验收**(RT/SubKS/KS 三指标齐看)、**不涨即回退**。
- 客户执行流程一律先进 `scripts/`,转发一行命令;产物放挂载盘;评测报告进 `logs/`;val 卷钉死。
- 观测三路:`metrics.jsonl` / TensorBoard / wandb;断点续跑;幂等闸;底座门禁(裸 SubKS<0.730 拒跑)。

---

## 10. 认证配置(密钥值不入库)

**Claude Code 走 Vertex**(本项目开发机的配置,可复制到其他机):
- `CLAUDE_CODE_USE_VERTEX=1`,`ANTHROPIC_VERTEX_PROJECT_ID=cloud-llm-preview1`,`CLOUD_ML_REGION=global`,`GOOGLE_APPLICATION_CREDENTIALS=<绝对路径>/vertex-key.json`。
- 认证根 = `leeway@google.com` 的 authorized_user 凭据文件(`vertex-key.json`)。**项目必须 `cloud-llm-preview1`、区域必须 `global`**(leeway-main 会 403,europe-west4 无模型)。
- settings.json 里路径必须**绝对路径**(Claude Code 不展开 `$HOME`)。

**Gemini 标注**(`rationalize_cot.py`):两条路,任选——
- express key(`AQ.` 前缀,走 `genai.Client(vertexai=True, api_key=...)`,不依赖 ADC);或
- ADC 走 Vertex:`genai.Client(vertexai=True, project="cloud-llm-preview1", location="global")` + `leeway@google.com` ADC。

> ⚠️ **所有密钥值(GitHub token、Gemini express key、refresh_token、client_secret)不写进本仓库。** 需要时走私密渠道获取/在本地凭据文件中读取。凭据用后建议轮换。
