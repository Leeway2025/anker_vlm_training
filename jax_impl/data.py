"""JAX 路线数据管线(独立实现,零 torch 依赖)。

样本 = 模板 ids(HF 基准排布,Gate B 已逐位对齐)+ label ids + 权重:
  - 模板来自 poc/02a 导出的 hf_layout.json(生产 prompt + 16 帧排布),
    视频占位 258884 → 哨兵 -2(Gate C 配方)
  - label: "{RT}|{SubKS}|{desc}"(无空格,与 GT 逐字节一致);
    分类段 ×4,desc ×1,think 段 ×0
    (与 torch 侧 loss 设计一致)
  - 固定 padding 到 max_len(XLA 静态形状)
支持: hard-mining 物理复制 / implicit-CoT(比例混合+退火)/ aux 标签。
帧: euno-wds 分片直读(tar 内 {video_id}.pyd pickle,16×JPEG bytes)。
"""
import dataclasses
import io
import json
import os
import pickle
import random
import re
import sys
import tarfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.taxonomy import AUX_HEAD_ORDER, aux_label_index, KS_GROUP, KS_CLASSES  # noqa: E402

EOT = 106            # <end_of_turn>
SENTINEL = -2        # 视觉哨兵(gemma4 JAX 模型的 TOKEN_PLACEHOLDER)


# ============================ dynseg(env 门控)============================
# TOKEN_COMPRESS_MODE=dynseg:保全部 n 帧、固定总预算 DYNSEG_TOTAL(默认512),
# 按逐帧像素活跃度动态分配预算(每帧∈[DYNSEG_FLOOR,64],和恒=TOTAL)。
# 宿主侧算逐帧预算(_dynseg_budget)并逐样本建变长模板;设备侧在
# _compress_soft_tokens 的 dynseg 分支按分数逐帧选 counts[i] 个、静态形状打包。
# 默认关:mode!=dynseg 时以下全部不触发,现有 topk/hyb/hyb2/dyn 路径零改动。

def _apportion(p, M, caps):
    """带上限的整数配额(最大余数法)。p:权重[n];M:待分配总量;caps:每项上限[n]。
    返回 int[n],sum==M,每项<=caps[i](要求 sum(caps)>=M,由调用方保证)。"""
    p = np.asarray(p, np.float64)
    caps = np.asarray(caps, np.int64)
    n = len(p)
    if p.sum() <= 0:
        p = np.ones(n, np.float64)
    quota = p / p.sum() * M
    alloc = np.minimum(np.floor(quota).astype(np.int64), caps)
    frac = quota - np.floor(quota)
    order = list(np.argsort(-frac))          # 余数大者优先
    short = int(M - alloc.sum())
    i = guard = 0
    while short > 0:
        idx = order[i % n]
        if alloc[idx] < caps[idx]:
            alloc[idx] += 1
            short -= 1
        i += 1
        guard += 1
        if guard > 100 * (n + M + 1):        # 兜底(sum(caps)>=M 时不会触发)
            break
    return alloc


def _dynseg_budget(frames, total, floor=8, cap=64, w_spatial=0.8):
    """逐帧动态视觉预算(纯 numpy,可 CPU 单测、可整体替换)。
    活跃度 = w_spatial·空间高频能量 + (1-w_spatial)·时序残差:
      (a) 空间高频 = 灰度帧有限差分梯度(|Δx|+|Δy|)均值 —— 对全局平移/云台
          pan 近似不变(平移不改梯度分布),故全局机位运动不会虚高;
      (b) 时序残差 = 相邻帧灰度差均值(归一化)—— 前景运动敏感,但权重更小。
    预算:counts = floor + apportion(softmax(活跃度), total-n·floor, cap-floor),
    整数最大余数法闭合、clamp 到 cap,sum==total 严格成立。返回 int32[n]。"""
    n = len(frames)
    assert n >= 1, "dynseg 至少 1 帧"
    assert floor * n <= total <= cap * n, (
        f"dynseg 预算不可行: total={total} n={n} floor={floor} cap={cap} "
        f"(需 floor*n<=total<=cap*n)")
    grays = []
    for f in frames:
        a = np.asarray(f).astype(np.float32)
        grays.append(a.mean(axis=-1) if a.ndim == 3 else a)
    spatial = np.empty(n, np.float64)
    for i, g in enumerate(grays):
        gx = np.abs(np.diff(g, axis=1)).mean() if g.shape[-1] > 1 else 0.0
        gy = np.abs(np.diff(g, axis=0)).mean() if g.shape[0] > 1 else 0.0
        spatial[i] = 0.5 * (gx + gy)
    temporal = np.zeros(n, np.float64)
    for i in range(1, n):
        temporal[i] = np.abs(grays[i] - grays[i - 1]).mean()
    if n > 1:
        temporal[0] = temporal[1]            # 首帧无前帧,借第二帧代理

    def _unit(x):
        s = float(x.sum())
        return x / s if s > 1e-12 else np.full(len(x), 1.0 / len(x))

    activity = w_spatial * _unit(spatial) + (1.0 - w_spatial) * _unit(temporal)
    a = activity - activity.max()            # softmax(去均值稳数值)
    p = np.exp(a)
    p /= p.sum()
    extra = _apportion(p, total - n * floor, np.full(n, cap - floor, np.int64))
    counts = (extra + floor).astype(np.int32)
    assert int(counts.sum()) == total
    assert counts.min() >= floor and counts.max() <= cap
    return counts


def _dynseg_template(head, delim, counts, tail, sentinel=SENTINEL):
    """逐样本变长模板:head + [S]*c0 + delim + [S]*c1 + ... + [S]*c_{n-1} + tail。
    sum(counts) 恒=TOTAL、delim/head/tail 固定 → 序列总长 T 跨样本恒定。"""
    seq = list(head)
    for i, c in enumerate(counts):
        if i > 0:
            seq.extend(delim)
        seq.extend([sentinel] * int(c))
    seq.extend(tail)
    return seq


_DYN_VI_CLS = None


def _dyn_vi_cls():
    """惰性定义并注册 DynVisionInput(PreprocessedVisionInput + seg_counts 数据域)。
    seg_counts 作为 pytree data_field → 逐样本逐帧预算以【数据】流经 jit(不是
    静态常量,故不会每批重编译、不会被烘死成首批的值)。仅 import gemma 时惰性
    执行 —— 保持 data.py 顶层 import 轻量、不初始化 jax/TPU。"""
    global _DYN_VI_CLS
    if _DYN_VI_CLS is None:
        import jax
        from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput

        @dataclasses.dataclass(frozen=True)
        class DynVisionInput(PreprocessedVisionInput):
            seg_counts: object = None        # [B, n] int32,逐帧预算(数据侧)
            ext_score: object = None         # [B, n, C] 外部打分(归因显著性等)
            full_res: bool = False           # True=满分辨率 KD 老师:_selected 跳过压缩,返回全 n*mst token

        jax.tree_util.register_dataclass(
            DynVisionInput,
            data_fields=["patches", "positions_xy", "seg_counts", "ext_score"],
            meta_fields=["soft_token_counts", "full_res"])
        _DYN_VI_CLS = DynVisionInput
    return _DYN_VI_CLS


def make_dyn_vision_input(patches, positions_xy, soft_token_counts, seg_counts):
    """构造 dynseg 用视觉输入(seg_counts 走数据侧到达 _selected)。"""
    cls = _dyn_vi_cls()
    return cls(patches=patches, positions_xy=positions_xy,
               soft_token_counts=soft_token_counts, seg_counts=seg_counts)


def make_full_vision_input(patches, positions_xy, soft_token_counts):
    """构造满分辨率视觉输入(full_res=True → _selected 跳过 _compress_soft_tokens,
    直接返回全部 n*mst 个 soft token)。仅用于满分辨率 KD 老师前向:老师看满 1024
    (16帧×64),学生仍压 512。student 侧 install_token_select 已把 _encode_vision
    monkeypatch 成 _selected,老师复用同一被 patch 的 Transformer 类,故 full_res
    分支生效。student 侧仍传普通 PreprocessedVisionInput(无 full_res 属性 → getattr
    默认 False),完全不受影响。soft_token_counts 在 full_res 分支未被使用(n 从
    patches 形状按 dyn 口径推得),传入仅为 meta 完整性。"""
    cls = _dyn_vi_cls()
    return cls(patches=patches, positions_xy=positions_xy,
               soft_token_counts=soft_token_counts, full_res=True)


def make_ext_vision_input(patches, positions_xy, soft_token_counts, ext_score):
    """构造带外部打分的视觉输入(ext_score 走数据侧到达 _compress_soft_tokens,
    覆盖运动/范数启发式打分 → 归因显著性 oracle 选择实验)。逐样本 ext_score 作为
    pytree data_field 流经 jit,不会被烘死成首样本常量。"""
    cls = _dyn_vi_cls()
    return cls(patches=patches, positions_xy=positions_xy,
               soft_token_counts=soft_token_counts, ext_score=ext_score)


def load_frames(rec, wds_dir):
    """分片定位优先级: meta.wds_dir(labels.jsonl 内声明,与 torch 侧
    euno_wds 行为一致)> 调用方传入的 wds_dir(--wds-dir / labels 同目录)。
    容器场景注意: meta.wds_dir 必须写容器内可见的路径 —— 最省事的做法是
    把分片目录以同名路径挂载(-v /真实路径:/真实路径),jsonl 零修改。"""
    meta = rec.get("meta") or {}
    base = meta.get("wds_dir") or wds_dir
    shard = os.path.join(base, f"shard-{meta.get('shard', 0):06d}.tar")
    with tarfile.open(shard) as tf:
        # 成员名约定与 torch 侧 euno_wds 一致: video_id 中的 "/" → "__"
        name = rec["video_id"].replace("/", "__") + ".pyd"
        try:
            raw = tf.extractfile(tf.getmember(name)).read()
        except KeyError:
            few = [m.name for m in tf.getmembers()[:3]]
            raise KeyError(f"{shard} 中无成员 {name!r};分片内实际成员形如 "
                           f"{few} —— 若命名约定不同请反馈")
    frames = pickle.loads(raw)["frames"]
    # env FRAME_SUBSAMPLE=N(默认0=关,取全部16帧):总预算不变、减帧路线——
    # 均匀抽 N 帧(时序覆盖满片),配 hf_layout_8f.json(512 视觉槽)+ SELECT_TOKENS_K=0。
    _fsub = int(os.environ.get("FRAME_SUBSAMPLE", "0"))
    if _fsub and _fsub < len(frames):
        idx = np.linspace(0, len(frames) - 1, _fsub).round().astype(int).tolist()
        frames = [frames[i] for i in idx]
    from PIL import Image
    return [np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
            for b in frames]


def load_jsonl_map(path):
    return {j["video_id"]: j for j in
            (json.loads(l) for l in open(path, encoding="utf-8"))}


def split_by_camera(recs, val_size, seed=0):
    """与 torch 侧 build_dataset.split_by_camera 同语义: 按摄像头整组
    进 val,防"记住这个门廊"式泄漏;无 camera_id / unknown 退化为按
    video_id。固定 seed → val 集跨运行稳定。"""
    rng = random.Random(seed)
    by_cam = {}
    for r in recs:
        cam = (r.get("meta") or {}).get("camera_id") or r["video_id"]
        if cam == "unknown":
            cam = r["video_id"]
        by_cam.setdefault(cam, []).append(r)
    cams = sorted(by_cam)
    rng.shuffle(cams)
    val, n = [], 0
    for c in cams:
        if n >= val_size:
            break
        val += by_cam[c]
        n += len(by_cam[c])
    val_ids = {r["video_id"] for r in val}
    return [r for r in recs if r["video_id"] not in val_ids], val


# desc 身份词表(方案三加权用): 判身份必须依赖的词
_ID_WORDS = re.compile(
    r"resident|home\s?owner|courier|delivery|staff|mail\s?carrier|"
    r"stranger|visitor|intruder|uniform|family|neighbor", re.I)


class SftDataset:
    def __init__(self, labels_file, layout_file, tokenizer, wds_dir=None,   # wds_dir 显式传入时覆盖 meta(见 load_frames)
                 max_label_len=64, cls_weight=4.0, rt_weight=0.0,
                 id_weight=0.0, sample_weights=None,
                 reasoning=None, cot_ratio=0.6, attributes=None,
                 max_think_len=96, seed=0, val_n=0,
                 aux_conf_threshold=0.5, augment=False, augment_v2=False,
                 augment_v3=False,
                 val_ids=None):
        recs = [json.loads(l) for l in open(labels_file, encoding="utf-8")]
        # 顺序铁律: 先切 val、再对 train 做 hard-mining 复制 —— 反过来
        # 副本会横跨 train/val(泄漏,val loss 虚低)。torch 侧同序。
        if val_ids:
            # 固定卷子: val 成员由外部清单钉死,训练数据增删不再改变
            # 选模标尺(消跑次方差中的"选模噪声";清单外的行全部进 train)
            val_recs = [r for r in recs if r["video_id"] in val_ids]
            train_recs = [r for r in recs if r["video_id"] not in val_ids]
            print(f"[data] val 固定卷子: 命中 {len(val_recs)}/{len(val_ids)}"
                  f"(清单内不在本数据文件的 {len(val_ids)-len(val_recs)} 条忽略)")
        elif val_n:
            train_recs, val_recs = split_by_camera(recs, val_n, seed=seed)
        else:
            train_recs, val_recs = recs, []
        self.loss_scale = {}    # vid → (0,1) 降权(噪声软化, 见下)
        if sample_weights:
            # 双语义(08-04): w>1 = hard-mining 物理复制(原有);
            # 0<w<1 = 嫌疑样本 loss 降权(不复制、不删除, __getitem__ 里
            # 乘进 per-token weights)—— judge 单证嫌疑"宁软勿删"用,
            # 保留样本信息量又不让可疑标签全额发声
            out = []            # 复制用流式最大余数法(round 会把
            acc = 0.0           # 1.0<w<1.5 全截成 1,类上限失效)
            for r in train_recs:
                w0 = float(sample_weights.get(r["video_id"], 1.0))
                if 0.0 < w0 < 1.0:
                    self.loss_scale[r["video_id"]] = w0
                w = max(1.0, w0)
                n = int(w)
                acc += w - n
                if acc >= 1.0:
                    n += 1
                    acc -= 1.0
                out.extend([r] * n)
            if len(out) != len(train_recs):
                print(f"[hard-mining] train {len(train_recs)} -> {len(out)}")
            if self.loss_scale:
                print(f"[soft-weight] 降权样本 {len(self.loss_scale)}")
            train_recs = out
        self.recs = train_recs + val_recs
        self.first_val = len(train_recs)        # ≥此下标 = val(无 CoT 注入)
        self.train_idx = list(range(len(train_recs)))
        self.val_idx = list(range(len(train_recs), len(self.recs)))
        self.wds_override = wds_dir            # 显式指定则最高优先
        self.wds = wds_dir or os.path.dirname(labels_file)
        lay = json.load(open(layout_file, encoding="utf-8"))
        self.template = [(SENTINEL if m == 2 else t) for t, m in
                         zip(lay["input_ids"], lay["mm_token_type_ids"])]
        # dynseg(env 门控,默认关):从基准排布分解 head/帧内视觉块/帧间delim/tail,
        # 逐样本按预算重建变长模板。分解源 = layout 的 vision_blocks(哨兵替换后
        # head/delim/tail 均无 mm==2 位,故与原始 ids 逐位一致)。
        self.seg_mode = os.environ.get("TOKEN_COMPRESS_MODE") == "dynseg"
        self.seg_total = int(os.environ.get("DYNSEG_TOTAL", "512"))
        self.seg_floor = int(os.environ.get("DYNSEG_FLOOR", "8"))
        if self.seg_mode:
            vb = lay["vision_blocks"]
            s0, c0 = vb[0]
            self._seg_head = list(self.template[:s0])
            self._seg_delim = (list(self.template[s0 + c0: vb[1][0]])
                               if len(vb) > 1 else [])
            ls, lc = vb[-1]
            self._seg_tail = list(self.template[ls + lc:])
            assert SENTINEL not in self._seg_head + self._seg_delim \
                + self._seg_tail, "head/delim/tail 不应含视觉哨兵"
            _fsub = int(os.environ.get("FRAME_SUBSAMPLE", "0"))
            self.seg_n = _fsub if (0 < _fsub < len(vb)) else len(vb)
            self.seg_T = (len(self._seg_head) + self.seg_total
                          + (self.seg_n - 1) * len(self._seg_delim)
                          + len(self._seg_tail))
        # tome(env 门控,默认关):时空 ToMe 把 n 帧×64 源合并为单块 TOME_TOTAL,
        # 丢弃帧间 delim(与 dyn 同款单块排布)。模板 = head + [S]*TOTAL + tail,
        # T 恒定。合并全在设备侧确定式完成 → 无逐样本元数据(比 dynseg 更简单,
        # 不需 DynVisionInput/pytree 数据域)。默认关:mode!=tome 时全部不触发。
        self.tome_mode = os.environ.get("TOKEN_COMPRESS_MODE") == "tome"
        self.tome_total = int(os.environ.get("TOME_TOTAL", "512"))
        if self.tome_mode:
            vb = lay["vision_blocks"]
            s0, c0 = vb[0]
            self._tome_head = list(self.template[:s0])
            ls, lc = vb[-1]
            self._tome_tail = list(self.template[ls + lc:])
            assert SENTINEL not in self._tome_head + self._tome_tail, \
                "head/tail 不应含视觉哨兵"
            self.tome_T = (len(self._tome_head) + self.tome_total
                           + len(self._tome_tail))
        self.tok = tokenizer
        self.max_label_len = max_label_len
        self.cls_w = cls_weight
        # CAP_WEIGHT: caption(desc)段逐 token 损失权重,默认 1.0(行为不变)。
        # 设 0 → 只训 RT|SubKS 打分前缀(客户"只训前3字符"口径),把梯度
        # 全压到打分位、不再被 ~60 caption token 稀释(教训#10:val_loss≠SubKS)。
        self.cap_w = float(os.environ.get("CAP_WEIGHT", "1.0"))
        self.rt_w = rt_weight                   # >0 时 RT 字母位单独加权(④)
        self.id_w = id_weight                   # >0 时 desc 身份词加权(方案三:
        #   逼模型为写对 resident/courier 等词先编码身份特征,不动判决先验)
        self.reasoning = reasoning or {}        # video_id → 资产 C
        self.cot_ratio = cot_ratio
        self.anneal = False                     # True → 纯生产模式
        self.max_think = max_think_len if self.reasoning else 0
        self.attributes = attributes or {}      # video_id → 资产 A
        self.aux_conf = aux_conf_threshold
        # 资产覆盖率横幅: 开跑即自检喂对了哪份资产(全量≈100%/白名单≈37%)
        if self.attributes or self.reasoning:
            tr_ids = {r["video_id"] for r in train_recs}
            for tag, m in (("aux(资产A)", self.attributes),
                           ("cot(资产C)", self.reasoning)):
                if m:
                    hit = len(tr_ids & set(m))
                    print(f"[assets] {tag}: 覆盖 train 独立视频 "
                          f"{hit}/{len(tr_ids)} ({hit/max(len(tr_ids),1):.1%})")
        self.rng = random.Random(seed)
        self.augment = augment                  # 仅 train 样本生效
        self.augment_v2 = augment_v2            # v2 三样叠加在 v1 之上
        self.augment_v3 = augment_v3            # v3 域定向,叠加在 v1/v2 之上
        self.max_len = len(self.template) + self.max_think + max_label_len
        if self.seg_mode:
            # dynseg 用 seg_T(远小于16帧模板)定长;末尾再留 seg_n 槽,
            # 把逐帧预算 seg_counts 编进 tokens 尾部(因果掩码之后、label/weight
            # 屏蔽区)——训练侧 loss_fn 无需改 shard_map 签名即可取到逐样本预算
            # 作为【数据】(见 train_sft.py dynseg 分支)。该尾区永不参与 loss。
            self.seg_off = self.seg_T + self.max_think + max_label_len
            self.max_len = self.seg_off + self.seg_n
        if self.tome_mode:
            # tome 用单块 tome_T(=head+TOTAL+tail)定长;无逐样本元数据尾区。
            self.max_len = self.tome_T + self.max_think + max_label_len

    def _augment(self, frames):
        """训练增强(语义移植自 torch data/augmentation.py,红线同款):
        ✓ 全片一致水平翻转 / 亮度缩放 / 帧 dropout(用前一帧顶替,
          时序保持非递减 —— 结构上不可能发生时序翻转)
        ✗ 时序翻转/mixup: 不提供任何实现路径(k↔l 等标签会互换)。"""
        r = self.rng
        if r.random() < 0.5:                       # 水平翻转(全片一致)
            frames = [np.ascontiguousarray(f[:, ::-1]) for f in frames]
        if r.random() < 0.5:                       # 亮度 ±25%
            k = r.uniform(0.75, 1.25)
            frames = [np.clip(f.astype(np.float32) * k, 0, 255)
                      .astype(np.uint8) for f in frames]
        if r.random() < 0.5:                       # 帧 dropout 10%
            out = [frames[0]]
            for f in frames[1:]:
                out.append(out[-1] if r.random() < 0.1 else f)
            frames = out
        return frames

    def _augment_v2(self, frames):
        """增强包 v2(默认关;必须在修正后干净 100k 上 from-scratch 消融
        验过才进配方 —— 慢显性手段,增量续训验证必假阴性):
        ✓ crop-zoom(全片同一裁剪框,放大近景身份细节;裁剪发生在
          preprocess 拉伸 384² 之前,与"RandomCrop 必须在 resize 前"
          的生产口径一致)
        ✓ 对比度缩放(全片同 k,围绕各帧自身均值,与 v1 亮度正交)
        ✓ 轻遮挡(全片同位置灰色矩形,模拟镜头污损/蛛网,逼模型用
          剩余区域证据 —— 位置跨帧固定,不破坏时序)
        ✗ 时序翻转/mixup/TTA: 与 v1 同款红线,不提供实现路径。"""
        r = self.rng
        h, w = frames[0].shape[:2]
        if r.random() < 0.5 and min(h, w) >= 32:   # crop-zoom(裁 70~100%)
            s = r.uniform(0.7, 1.0)
            ch, cw = max(int(h * s), 16), max(int(w * s), 16)
            y0, x0 = r.randint(0, h - ch), r.randint(0, w - cw)
            frames = [np.ascontiguousarray(f[y0:y0 + ch, x0:x0 + cw])
                      for f in frames]
        if r.random() < 0.5:                       # 对比度 ±25%
            k = r.uniform(0.75, 1.25)
            frames = [np.clip((f.astype(np.float32) - f.mean()) * k
                              + f.mean(), 0, 255).astype(np.uint8)
                      for f in frames]
        if r.random() < 0.3:                       # 轻遮挡(每边 10~30%)
            h, w = frames[0].shape[:2]
            oh, ow = int(h * r.uniform(0.1, 0.3)), int(w * r.uniform(0.1, 0.3))
            y0, x0 = r.randint(0, h - oh), r.randint(0, w - ow)
            frames = [f.copy() for f in frames]
            for f in frames:
                f[y0:y0 + oh, x0:x0 + ow] = 114
        return frames

    def _augment_v3(self, frames):
        """增强包 v3 — 域定向(用户 0811 07:46:攻具体域偏移,非通用不变性;
        必须在干净 100k from-scratch 消融验过才进配方 —— 与 v2 同红线:
        慢显性手段,增量续训验证必假阴性)。四样,全片一致(逐样本抽参):
        ✓ 时序连续窗:取 L∈[12,N] 连续帧,线性拉回 N 帧(定长保 counts 一致,
          不触 make_vision_input 断言;索引非递减,不翻时序)(p=0.5)
        ✓ 灰度/IR 模拟:RGB→亮度 + 轻暖/冷绿/纯灰偏,攻"日彩↔夜红外同语义"
          —— 同一事件不会同时有彩色版和红外版,真数据教不全(p=0.3)
        ✓ 低照度+传感器噪声:gamma 压暗(全片一致)+ 高斯噪声(逐帧独立=真实
          传感器),夜间样本的清晰版孪生(p=0.3)
        ✓ 桶形/鱼眼 warp:scipy map_coordinates 径向畸变(系数全片一致,coords
          预算一次),攻广角边缘拉伸的人/车不变性(p=0.2)
        ✗ 时序翻转/跨类 mixup:同 v1/v2 红线,不提供实现路径。"""
        r = self.rng
        N = len(frames)
        # ① 时序连续窗(先做:后续光度/几何作用在窗上)
        if N >= 12 and r.random() < 0.5:
            L = r.randint(12, N)
            s = r.randint(0, N - L)
            win = frames[s:s + L]
            idx = [min(L - 1, int(round(t * (L - 1) / (N - 1))))
                   for t in range(N)]          # 线性拉回 N 帧,非递减
            frames = [win[k] for k in idx]
        # ② 灰度 / IR 模拟
        if r.random() < 0.3:
            tint = r.choice([(1.00, 0.98, 0.94),   # 暖(白炽/近红外)
                             (0.94, 1.00, 0.96),   # 冷绿(夜视仪)
                             (1.00, 1.00, 1.00)])  # 纯灰
            lum = np.array([0.299, 0.587, 0.114], np.float32)
            out = []
            for f in frames:
                g = f.astype(np.float32) @ lum
                g3 = np.stack([g * tint[0], g * tint[1], g * tint[2]], -1)
                out.append(np.clip(g3, 0, 255).astype(np.uint8))
            frames = out
        # ③ 低照度 + 传感器噪声
        if r.random() < 0.3:
            gamma = r.uniform(1.8, 3.0)            # 压暗(全片一致)
            gain = r.uniform(0.6, 1.0)
            sigma = r.uniform(4.0, 14.0)
            lut = np.clip(((np.arange(256) / 255.0) ** gamma) * gain * 255.0,
                          0, 255).astype(np.uint8)
            npr = np.random.default_rng(r.randrange(1 << 31))
            out = []
            for f in frames:                       # 噪声逐帧独立(真实传感器)
                d = lut[f].astype(np.float32) + npr.normal(0, sigma, f.shape)
                out.append(np.clip(d, 0, 255).astype(np.uint8))
            frames = out
        # ④ 桶形/鱼眼 warp(系数全片一致,坐标图预算一次)
        if r.random() < 0.2:
            from scipy.ndimage import map_coordinates
            h, w = frames[0].shape[:2]
            k = r.uniform(0.10, 0.35)
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            ny, nx = (yy - cy) / max(cy, 1), (xx - cx) / max(cx, 1)
            f2 = 1.0 + k * (nx * nx + ny * ny)     # 径向系数
            coords = np.stack([ny * f2 * cy + cy, nx * f2 * cx + cx])
            out = []
            for f in frames:
                chans = [map_coordinates(f[..., c], coords, order=1,
                                         mode="nearest") for c in range(3)]
                out.append(np.stack(chans, -1).astype(np.uint8))
            frames = out
        return [np.ascontiguousarray(f) for f in frames]

    def set_anneal(self, flag):                 # CoT 退火(torch 同款语义)
        self.anneal = flag

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        lb = rec["labels"]
        vid = rec["video_id"]
        # 目标与 GT 逐字节一致: "{RT}|{SK}|{desc}"(无空格;旧版 " | "
        # 与客户生产口径不符,跨框架续训 token 序列冲突 —— B4 修复)。
        # 整串单次分词: 无空格交界会并子词,分段 encode 拼接≠真实序列;
        # 权重按字符覆盖回填(与 torch offset 语义一致): 与 cls 前缀
        # 有重叠的 token ×cls_w,其余 ×1
        cls_txt = f"{lb['role_type']}|{lb['sub_keyscene']}|"
        full_txt = cls_txt + str(lb["description"]).strip()
        tgt_ids = self.tok.encode(full_txt)
        id_spans = ([(m.start(), m.end()) for m in _ID_WORDS.finditer(full_txt)]
                    if self.id_w else [])
        tgt_w, start = [], 0
        for k in range(1, len(tgt_ids) + 1):   # 注意勿用 i(样本下标)
            end = len(self.tok.decode(tgt_ids[:k]))
            if self.rt_w and start < 2:   # 首 2 字符 = RT 字母+竖线
                tgt_w.append(self.rt_w)
            elif start < len(cls_txt):
                tgt_w.append(self.cls_w)
            elif any(s0 < end and start < e0 for s0, e0 in id_spans):
                tgt_w.append(self.id_w)   # desc 身份词 token
            else:
                tgt_w.append(self.cap_w)  # caption 段(CAP_WEIGHT,默认 1.0)
            start = end

        think_ids = []
        # val 样本(i >= first_val)永不注入 CoT: 保证 val loss 分布固定、
        # 各次 eval 可比 —— 否则 best checkpoint 选择近似掷骰子
        if (i < self.first_val and self.reasoning.get(vid)
                and not self.anneal
                and self.rng.random() < self.cot_ratio):
            r = self.reasoning[vid]
            txt = r.get("reasoning_chain") or (
                f"[Identity cues] {r.get('identity_clues', '')} "
                f"[Scene cues] {r.get('scene_clues', '')} "
                f"[Conclusion] {r.get('conclusion', '')}")
            think_ids = self.tok.encode(txt)[: self.max_think]

        lab = list(think_ids) + list(tgt_ids) + [EOT]
        w = ([0.0] * len(think_ids)             # think 段权重 0(隐式 CoT)
             + tgt_w + [1.0])
        cap = self.max_think + self.max_label_len
        lab, w = lab[:cap], w[:cap]
        sc = self.loss_scale.get(vid)
        if sc and i < self.first_val:   # 噪声软化: 仅 train, val 不动
            w = [x * sc for x in w]

        # 辅助头标签: 资产 A 的 7 属性 + KS 父类(6 类)
        attrs = self.attributes.get(vid, {})
        av = attrs.get("attributes", attrs)
        # 置信度门(对齐 torch): 低置信 Gemini 标注整条屏蔽,
        # 否则辅助头在带噪标签上训练
        low_conf = float(attrs.get("confidence", 1.0)) < self.aux_conf
        aux = np.array([aux_label_index(h, av.get(h, ""))
                        if (av and not low_conf) else -100
                        for h in AUX_HEAD_ORDER], np.int32)
        ks = KS_CLASSES.index(KS_GROUP[lb["sub_keyscene"]]) \
            if lb["sub_keyscene"] in KS_GROUP else -100

        # 优先级: 显式 wds_dir(--wds-dir)> meta.wds_dir > labels 同目录
        # (帧须先于建 token:dynseg 逐样本模板依赖对增强后帧算的逐帧预算。
        #  RNG 时序不变:CoT 注入用 rng 在前、增强用 rng 在后,建 token/aux
        #  不用 rng,故移动建 token 位置不改变随机序列 → 现有模式行为零变化。)
        if self.wds_override:
            frames = load_frames({**rec, "meta": {**(rec.get("meta") or {}),
                                                  "wds_dir": self.wds_override}},
                                 self.wds)
        else:
            frames = load_frames(rec, self.wds)
        if i < self.first_val:
            if self.augment:
                frames = self._augment(frames)
            if self.augment_v2:
                frames = self._augment_v2(frames)
            if self.augment_v3:
                frames = self._augment_v3(frames)

        L = self.max_len
        tokens = np.zeros(L, np.int32)
        labels = np.full(L, -100, np.int32)
        weights = np.zeros(L, np.float32)
        seg_counts = None
        if self.seg_mode:
            n = len(frames)
            assert n == self.seg_n, (
                f"dynseg 帧数 {n} != 预期 {self.seg_n}(检查 FRAME_SUBSAMPLE)")
            seg_counts = _dynseg_budget(frames, self.seg_total, self.seg_floor)
            tmpl = _dynseg_template(self._seg_head, self._seg_delim,
                                    seg_counts, self._seg_tail)
            T = len(tmpl)
            assert T == self.seg_T, f"dynseg 模板长 {T} != 恒定 {self.seg_T}"
            tokens[:T] = tmpl
            # 逐帧预算编入 tokens 尾部保留区(因果掩码之后、label 屏蔽区,
            # 永不参与 loss/logits)→ 训练侧 loss_fn 取作数据侧 seg_counts
            tokens[self.seg_off:self.seg_off + n] = seg_counts
        elif self.tome_mode:
            # tome 单块模板:head + [S]*TOME_TOTAL + tail(丢帧间 delim)。
            tmpl = (self._tome_head + [SENTINEL] * self.tome_total
                    + self._tome_tail)
            T = len(tmpl)
            assert T == self.tome_T, f"tome 模板长 {T} != 恒定 {self.tome_T}"
            tokens[:T] = tmpl
        else:
            T = len(self.template)
            tokens[:T] = self.template
        tokens[T:T + len(lab)] = lab
        labels[T:T + len(lab)] = lab
        weights[T:T + len(lab)] = w

        out = {"tokens": tokens, "labels": labels, "weights": weights,
               "aux_labels": aux, "ks_label": np.int32(ks),
               "frames": frames, "video_id": vid}
        if seg_counts is not None:
            out["seg_counts"] = seg_counts
        return out


def make_vision_input(frames_list):
    """B 个样本(各 16 帧)→ 模型入参 [B, n*p, ·]。
    B>1 需先 install_batched_encode_vision()(poc/05 等价测试 PASS)。
    每帧 soft token 数由 env MAX_SOFT_TOKENS 控制(默认 64=旧行为);
    降它=帧内 patch 压缩(max_patches=MST*9,同时缩编码器 patch 数+LLM 入参),
    用于 token 压缩 Pareto(选/压 patch 降 input token,边端提速)。"""
    import os
    from gemma.gm.nn.gemma4.vision._preprocessing import preprocess_and_patchify
    mst = int(os.environ.get("MAX_SOFT_TOKENS", "64"))
    sel_k = int(os.environ.get("SELECT_TOKENS_K", "0"))
    seg_mode = os.environ.get("TOKEN_COMPRESS_MODE") == "dynseg"
    tome_mode = os.environ.get("TOKEN_COMPRESS_MODE") == "tome"
    tome_total = int(os.environ.get("TOME_TOTAL", "512"))
    rsp_mode = os.environ.get("TOKEN_RESAMPLER", "0") == "1"
    rsp_total = int(os.environ.get("RESAMPLER_TOKENS", "512"))
    pas, poss, counts0 = [], [], None
    for frames in frames_list:
        patches, pos, counts = preprocess_and_patchify(
            frames, max_soft_tokens=mst)
        n, p, d = patches.shape
        pas.append(patches.reshape(n * p, d))
        poss.append(pos.reshape(n * p, 2))
        counts = tuple(int(c) for c in counts)
        if seg_mode:
            # dynseg:全帧满 mst 送编码器,真实逐帧预算走数据侧 seg_counts,
            # 设备侧在 _compress_soft_tokens 的 dynseg 分支静态形状选出 TOTAL。
            # soft_token_counts 保持逐样本一致的静态元组(=每帧 mst,长度=帧数)。
            assert all(c == mst for c in counts), "dynseg 要求满帧正方形"
            counts = (mst,) * n
        elif tome_mode:
            # tome:全帧满 mst 送编码器,设备侧时空双部软匹配静态合并成 TOME_TOTAL,
            # 打包成单块(帧数在补丁内由 patch 总数反推,与 dyn 同);文本侧占位=TOTAL。
            assert all(c == mst for c in counts), "tome 要求满帧正方形"
            counts = (tome_total,)
        elif rsp_mode:
            # 学习式重采样器(env TOKEN_RESAMPLER=1,默认关):RESAMPLER_TOKENS
            # 个可学习 query 对全帧 soft token 做 cross-attention,输出固定
            # [B, TOTAL, D] —— 用"软聚合"替换 topk/dyn 的硬选择(信息不丢弃、
            # 端到端可导,无需 STE)。counts 口径与 mode=dyn 单块同款:(TOTAL,)
            # 单块,文本模板零改动(merge 只消费前 TOTAL 个哨兵位)。
            assert all(c == mst for c in counts), "resampler 要求满帧正方形"
            assert sel_k == 0, ("TOKEN_RESAMPLER=1 与 SELECT_TOKENS_K 互斥"
                                "(重采样器整体替换硬选择,勿叠加)")
            counts = (rsp_total,)
        elif sel_k > 0:
            # 池化后 token 压缩:文本侧占位数对齐压缩后 token 数;真实压缩在
            # install_token_select() 补丁内做(需全帧 soft token 打分)。
            # mode=topk: 逐帧留 K;mode=hyb: 帧0 全保(关键帧)+ 其余帧各 K。
            assert all(c == mst for c in counts), "token 压缩要求满帧正方形"
            assert sel_k < mst, "SELECT_TOKENS_K 必须 < MAX_SOFT_TOKENS"
            mode = os.environ.get("TOKEN_COMPRESS_MODE", "topk")
            if mode in ("hyb", "hyb2"):
                counts = (mst,) + (sel_k,) * (n - 1)
            elif mode == "dyn":
                counts = (n * sel_k,)   # 全局竞争打包成单块(帧数在补丁内重推)
            else:
                counts = (sel_k,) * n
        assert counts0 in (None, counts), "batch 内 counts 必须一致"
        counts0 = counts
    if sel_k > 0 or seg_mode or tome_mode or rsp_mode:
        install_token_select()
    return (np.stack(pas), np.stack(poss), counts0)


def _compress_soft_tokens(t, k, mode, seg=None, total=None,
                          head=None, training=False, ext_score=None):
    """池化后 soft token 压缩核心(纯函数,可 CPU 单测)。
    t: [B, n, C, D] 全帧 soft token → [B, total, D]。
    打分:时序显著性(与该网格位置跨帧均值的 L2;固定机位下重复背景≈0,
    运动前景高分)z-score + 0.5×特征范数 z-score;选中索引 sort 保光栅序。
    mode=dynseg: 逐帧变预算(seg[B,n],和=total);逐帧按分数选 seg[b,i] 个,
      静态形状打包成 [B, total, D](帧序、帧内光栅序)。seg 走数据侧。
    mode=topk: 逐帧 Top-K 硬选(total=n*K)。
    mode=hyb : 2+3+4 合一(total=C+(n-1)*K)——
      帧0 关键帧全保(跨帧冗余的"那一份",方法2);
      帧1.. 逐帧 Top-K(方法4);
      非选中不丢、按余弦就近合并进选中 token=簇均值(方法3/PruMerge)。"""
    import jax.numpy as jnp
    from jax import lax, nn as jnn
    B, n, C, D = t.shape
    motion = jnp.linalg.norm(t - jnp.mean(t, axis=1, keepdims=True), axis=-1)
    norm = jnp.linalg.norm(t, axis=-1)

    def z(x):
        return (x - jnp.mean(x, axis=-1, keepdims=True)) / (
            jnp.std(x, axis=-1, keepdims=True) + 1e-6)
    score = z(motion) + 0.5 * z(norm)                   # [B, n, C]
    if os.environ.get("TOKEN_ATTNPROXY", "0") == "1":
        # ⑤b 被注意度(0814):对每帧 64 个池化 token 算 softmax(HH^T/√D)
        # 的列和=被全帧"指向"的程度(塔内注意力的零手术事后代理,语义显著
        # 物体特征独特被指向多)。64×64 矩阵乘,NPU 无压力。
        tn_ = t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + 1e-6)
        att_ = jnn.softmax(
            jnp.einsum("bncd,bnkd->bnck", tn_, tn_) * (D ** 0.5) / 8.0,
            axis=-1)
        score = score + 0.7 * z(jnp.sum(att_, axis=2))  # 列和=被注意度
    if os.environ.get("TOKEN_DILATE", "0") == "1":
        # 邻域增强(0814 g类归因:按门铃=小运动小面积,手/门铃token在全局
        # 排序中落选)。把 8×8 网格上 3×3 max 池化的 0.6 倍与原分取 max——
        # 高分核心的空间邻居(臂旁的手、手旁的门铃)被抬进 Top-K。纯静态。
        _g = int(round(C ** 0.5))
        _sg = jnp.reshape(score, (B, n, _g, _g))
        _sm = lax.reduce_window(_sg, -jnp.inf, lax.max,
                                (1, 1, 3, 3), (1, 1, 1, 1), "SAME")
        score = jnp.maximum(score, 0.6 * jnp.reshape(_sm, (B, n, C)))
    if os.environ.get("TOKEN_LEARN_SCORE", "0") == "1" and head is not None:
        # ⑥ 可学习打分头(env TOKEN_LEARN_SCORE,默认关):低秩头读池化 soft
        # token 特征给出附加分 s = tanh(t@A)@b(A[D,r]、b[r,1]),加进选择分数。
        # warm-start: b=0 → s≡0 → 选择与关闭时逐位一致(零冲击暖启)。梯度经
        # A/b 回流(参数由 install_token_select 的 patched setup 用 self.param
        # 注册,进 params pytree;硬 top_k 不可导 → 见下方 training STE 软门)。
        A_h, b_h = head
        s = jnp.tanh(jnp.einsum("bncd,dr->bncr", t, A_h))
        s = jnp.einsum("bncr,ro->bnco", s, b_h)[..., 0]     # [B, n, C]
        # 不再 z-归一化 s:warm-start b=0→s≡0→选择逐位不变;直接(带增益)相加。
        # 不用 z(s):z 会把近零信号强行放大~1/std→选择被瞬间主导,且 std=0 处
        # d(std)=1/std→梯度 NaN 首步毁参数。改用固定增益 gain 给头一个『量纲』——
        # 底座打分是 z-score 后 O(1),而 s 原始量级 ~O(0.01·|b|) 太小压不动选择;
        # gain 把头的影响力抬到可与底座竞争的尺度(纯乘、无奇点、warm-start b=0→0
        # 不变;梯度同步×gain 让头学得更快)。TOKEN_LEARN_GAIN 默认 1.0=旧行为。
        gain = float(os.environ.get("TOKEN_LEARN_GAIN", "1.0"))
        score = score + gain * s
    if ext_score is not None:
        # 归因显著性 oracle:外部打分 [B,n,C] 完全覆盖运动/范数启发式(数据侧
        # pytree 流入,逐样本正确)。高分=保留(topk/dyn 选最高分)。
        score = jnp.asarray(ext_score).astype(score.dtype)
    # STE 软门温度(仅 training=True 的 topk/dyn 分支用;env 门控默认 0.5)
    temp = float(os.environ.get("TOKEN_STE_TEMP", "0.5"))

    def _ste_mult(g_all, sel_idx, axis):
        # 直通估计器(STE)乘子: 前向 == 1.0(硬选逐位不变),反向携带
        # d g_sel → d score → d(A,b)。g_sel/停梯(g_sel) 恒 1;where 防 0/0。
        g_sel = jnp.take_along_axis(g_all, sel_idx, axis=axis)
        denom = lax.stop_gradient(g_sel)
        return g_sel / jnp.where(denom > 0, denom, 1.0)

    if mode == "dynseg":
        # 逐帧变预算 seg[B,n](和=total,∈[floor,C]),静态形状选出总 total 个:
        #   ① 帧内名次 rank(0=最高分),keep=rank<seg → [B,n,C],逐帧和=seg[b,i]
        #   ② 展平后 val = keep*BIG - flat_idx;top_k(val,total) 恰取全部保留者,
        #      且按 flat_idx 升序(=帧序、帧内光栅序)返回;再 sort 稳固该序
        # 全程静态形状(top_k total、argsort C 均编译期常量)→ NPU 友好。
        assert seg is not None and total is not None, "dynseg 需 seg 与 total"
        order = jnp.argsort(-score, axis=-1)             # [B,n,C] 分数降序位置
        rank = jnp.argsort(order, axis=-1)               # 逆置换 = 每 token 名次
        keep = (rank < seg[..., None]).astype(jnp.int32)  # [B,n,C] 逐帧和=seg
        keep_flat = jnp.reshape(keep, (B, n * C))
        raster = jnp.arange(n * C)[None, :]              # 展平索引=帧*C+帧内光栅
        big = n * C + 1
        val = keep_flat * big - raster
        _, sidx = lax.top_k(val, total)                  # total 个保留者
        sidx = jnp.sort(sidx, axis=-1)                   # 复原 (帧,光栅) 序
        tf = jnp.reshape(t, (B, n * C, D))
        return jnp.take_along_axis(tf, sidx[..., None], axis=1)  # [B,total,D]
    if mode == "tome":
        # 时空 ToMe(双部软匹配,env 门控,默认关):把 n 帧×C 个 soft token
        # 展平成 N=n*C 的全局序列,静态压到 total 个,输出 [B, total, D]。
        #   ① 双部切分:按展平(帧,光栅)序偶/奇位分成 A、B 两集(各 N/2)。
        #      跨帧冗余(固定机位下静态背景在多帧几乎相同)天然分落 A/B,故
        #      A 里某帧的 token 可与 B 里【另一帧】的相似 token 合并 —— 空间×时间。
        #   ② 相似度:A、B 各 L2 归一后一次 batched matmul 得 [B,Na,Nb] 余弦;
        #      每个 A token 取最相似的 B token(argmax)= 其归并目标 dst。
        #   ③ 合并数 r=Na-n_keep(n_keep=total-Nb):把 r 个"最冗余"(edge 最高)
        #      的 A 并入各自 dst,余 n_keep 个"最独特"(edge 最低)的 A 原样保留。
        #      total==Nb(如 1024→512)时 n_keep=0 → 全并、无 top_k(最省、最好导出)。
        #   ④ 段均值:one-hot(dst)×merge_mask 做"scatter-mean"= 两次 matmul:
        #      merged[b,j] = (B[b,j] + Σ_{并入 j 的 A} A) / (1 + 计数)。
        # 全程静态形状(reshape/matmul/argmax/one_hot;仅 n_keep>0 才用 top_k)
        # → NPU/RKNN 友好(见 outputs/tome_scope.md)。
        assert total is not None, "tome 需 total"
        tf = jnp.reshape(t, (B, n * C, D))
        N = n * C
        assert N % 2 == 0, f"tome 需偶数源 token 数,得 N={N}"
        half = N // 2
        pair = jnp.reshape(tf, (B, half, 2, D))
        A = pair[:, :, 0, :]                                 # [B, half, D] 偶位
        Bs = pair[:, :, 1, :]                                # [B, half, D] 奇位
        Na = Nb = half
        n_keep = total - Nb
        assert 0 <= n_keep <= Na, (
            f"tome 预算不可行: total={total} Nb={Nb} Na={Na}(需 Nb<=total<=N)")
        r = Na - n_keep                                      # 待合并的 A 数
        An = A / (jnp.linalg.norm(A, axis=-1, keepdims=True) + 1e-6)
        Bn = Bs / (jnp.linalg.norm(Bs, axis=-1, keepdims=True) + 1e-6)
        sim = jnp.einsum("bad,bnd->ban", An, Bn)             # [B, Na, Nb] 余弦
        dst = jnp.argmax(sim, axis=-1)                       # [B, Na] 最相似 B
        oh = jnn.one_hot(dst, Nb, dtype=t.dtype)             # [B, Na, Nb]
        keep_idx = None
        if n_keep > 0:
            edge = jnp.max(sim, axis=-1)                     # [B, Na] 匹配强度
            _, keep_idx = lax.top_k(-edge, n_keep)           # edge 最低=最独特
            keep_idx = jnp.sort(keep_idx, axis=-1)
            merge_mask = jnp.ones((B, Na), t.dtype).at[
                jnp.arange(B)[:, None], keep_idx].set(0.0)
            oh = oh * merge_mask[..., None]                  # 保留者不并入
        contrib = jnp.einsum("bad,ban->bnd", A, oh)          # [B, Nb, D] 段和
        count = jnp.sum(oh, axis=1)                          # [B, Nb] 段计数
        merged = (Bs + contrib) / (1.0 + count[..., None])   # [B, Nb, D] 段均值
        if n_keep > 0:
            keptA = jnp.take_along_axis(A, keep_idx[..., None], axis=1)
            return jnp.concatenate([merged, keptA], axis=1)  # [B, total, D]
        return merged                                        # [B, Nb==total, D]
    if mode == "topk":
        sv, idx = lax.top_k(score, k)
        idx = jnp.sort(idx, axis=-1)
        sel = jnp.take_along_axis(t, idx[..., None], axis=2)
        if training:
            # 逐帧第 k 大分数为软门阈 tau(stop_gradient),门 g=sigmoid((score
            # -tau)/temp);对选中 token 乘 STE 乘子 → 前向不变、梯度回打分头。
            tau = lax.stop_gradient(sv[..., -1:])           # [B, n, 1]
            g = jnn.sigmoid((score - tau) / temp)           # [B, n, C]
            sel = sel * _ste_mult(g, idx, axis=-1)[..., None]
        return jnp.reshape(sel, (B, n * k, D))
    if mode == "hyb":
        key = t[:, 0]                                   # [B, C, D] 帧0全保
        rest, rs = t[:, 1:], score[:, 1:]
        _, idx = lax.top_k(rs, k)
        idx = jnp.sort(idx, axis=-1)
        sel = jnp.take_along_axis(rest, idx[..., None], axis=2)  # [B,n-1,K,D]
        tn = rest / (jnp.linalg.norm(rest, axis=-1, keepdims=True) + 1e-6)
        sn = sel / (jnp.linalg.norm(sel, axis=-1, keepdims=True) + 1e-6)
        sim = jnp.einsum("bfcd,bfkd->bfck", tn, sn)     # 余弦相似
        oh = jnn.one_hot(jnp.argmax(sim, -1), k, dtype=t.dtype)  # 就近指派
        merged = jnp.einsum("bfcd,bfck->bfkd", rest, oh) / (
            jnp.swapaxes(jnp.sum(oh, axis=2, keepdims=True), 2, 3) + 1e-6)
        return jnp.concatenate(
            [key, jnp.reshape(merged, (B, (n - 1) * k, D))], axis=1)
    if mode == "hyb2":
        # hyb 三升级合体(0812 用户确认排队;预算与 hyb 严格同=C+(n-1)*k):
        #   ①打分参照改帧0(站定目标相对帧0永远高分,补徘徊盲区)
        #   ②摘要用余弦相似度加权(减垃圾污染)
        #   ③落选者不并入选中,聚成 ns 个独立摘要 token(前景零污染),
        #     每帧 = (k-ns) 个纯选中 + ns 摘要。
        ns = 4
        key = t[:, 0]
        rest = t[:, 1:]                                       # [B,n-1,C,D]
        sc = z(jnp.linalg.norm(rest - t[:, :1], axis=-1)) \
            + 0.5 * z(norm[:, 1:])                            # ①与帧0偏差
        _, idx = lax.top_k(sc, k - ns)
        idx = jnp.sort(idx, axis=-1)
        sel = jnp.take_along_axis(rest, idx[..., None], axis=2)   # ③纯保留
        bi = jnp.arange(B)[:, None, None]
        fi = jnp.arange(n - 1)[None, :, None]
        sc_lose = sc.at[bi, fi, idx].set(-jnp.inf)            # 选中者出局
        _, lidx = lax.top_k(sc_lose, C - (k - ns))            # 全部落选者
        lose = jnp.take_along_axis(rest, lidx[..., None], axis=2)
        seeds = lose[:, :, :ns]                               # 分最高的ns个当种子
        ln = lose / (jnp.linalg.norm(lose, axis=-1, keepdims=True) + 1e-6)
        sn2 = seeds / (jnp.linalg.norm(seeds, axis=-1, keepdims=True) + 1e-6)
        sim2 = jnp.einsum("bfcd,bfsd->bfcs", ln, sn2)         # 余弦
        assign = jnn.one_hot(jnp.argmax(sim2, -1), ns, dtype=t.dtype)
        w = assign * jnn.relu(jnp.max(sim2, -1, keepdims=True))  # ②相似度加权
        ctx = jnp.einsum("bfcd,bfcs->bfsd", lose, w) / (
            jnp.sum(w, axis=2)[..., None] + 1e-6)
        per = jnp.concatenate([sel, ctx], axis=2)             # [B,n-1,k,D]
        return jnp.concatenate(
            [key, jnp.reshape(per, (B, (n - 1) * k, D))], axis=1)
    assert mode == "dyn", mode
    # dyn = 总量固定 n*K、全局竞争分配(每帧保底 FLOOR,余额全池 Top 竞争;
    #   热闹帧自然多中选、空帧少中选;打包成单块,形状恒 [B, n*K, D])。
    floor = 8
    assert k > floor
    _, fidx = lax.top_k(score, floor)                   # [B, n, floor]
    off = (jnp.arange(n) * C)[None, :, None]
    floor_flat = jnp.reshape(fidx + off, (B, n * floor))
    flat = jnp.reshape(score, (B, n * C))
    masked = flat.at[jnp.arange(B)[:, None], floor_flat].set(-jnp.inf)
    _, gidx = lax.top_k(masked, n * k - n * floor)      # 余额全局竞争
    all_idx = jnp.sort(jnp.concatenate([floor_flat, gidx], axis=-1), axis=-1)
    tf = jnp.reshape(t, (B, n * C, D))
    out = jnp.take_along_axis(tf, all_idx[..., None], axis=1)
    if training:
        # dyn 全局竞争选中者的 STE 软门:阈 tau=逐帧第 k 大分数(stop_gradient),
        # g=sigmoid((score-tau)/temp) 展平后在 all_idx 处取门值 → 乘 STE 乘子
        # (前向逐位不变,反向把梯度送回打分头 A/b)。
        tau = lax.stop_gradient(lax.top_k(score, k)[0][..., -1:])   # [B, n, 1]
        g = jnp.reshape(jnn.sigmoid((score - tau) / temp), (B, n * C))
        out = out * _ste_mult(g, all_idx, axis=1)[..., None]
    return out


# ==================== 学习式重采样器(env TOKEN_RESAMPLER,默认关)====================
# TOKEN_RESAMPLER=1:用可学习重采样器整体替换硬选择 —— RESAMPLER_TOKENS(默认512)
# 个可学习 query 对 n 帧×C 个 soft token 做 RESAMPLER_LAYERS(默认1,可2)层
# cross-attention(可选 FFN),输出固定 [B, TOTAL, D]。设计动机:
#   ① 0814 已证"硬 Top-K 选哪些 token"接近 no-op、信息损失在【硬丢弃】本身
#      (学习打分头 +0.03 噪声级;oracle 选择≈运动启发式)—— 重采样=软聚合,
#      每个输出 token 是全部 1024 个源 token 的注意力加权和,不丢弃任何信息;
#   ② 端到端可导(softmax 注意力),无 top_k 不可导问题 → 不需要 STE 技巧;
#   ③ 全静态形状(matmul/softmax/reshape),NPU 可导出。
# counts 口径 = mode=dyn 单块同款:(TOTAL,) 单块;文本模板零改动(gm 的
# merge_flat_embeddings 用 nonzero(mask, size=TOTAL) 只消费前 TOTAL 个哨兵位,
# 与现有 dyn/topk 运行口径一致,实测见 prefill=1359 的 dyn 链路日志)。
# 参数通道 = 打分头 tok_scorer 同款(已验证可训练+可保存/加载):
#   install_token_select 的 patched setup 用 self.param 注册 → 进 params pytree
#   → train_sft 抽成 train["tok"] 子树(优化器 "tok" 组)→ npz 存 "tok/<路径>"
#   → infer 按 struct 真实路径注回 merged params。默认关:不建任何参数,
#   现有 topk/hyb/hyb2/dyn/dynseg/tome 路径零改动。

def _resampler_specs(D, total, layers, ffn, ffn_mult):
    """重采样器参数清单 [(name, shape, kind)] —— 单一事实源:
    setup 注册(install_token_select)、宿主初始化(resampler_init_leaf 的
    调用方)、热身脚本(resampler_warmup)都以此为准,防三处漂移。
    结构(每层):
      q' = q + MHA(LN_q(q), LN_kv(x) 作 k, 【原始 x】作 v)   # cross-attn
      q' = q' + FFN(LN2(q'))                    # 可选(RESAMPLER_FFN=1 默认开)
    v 路径读【原始 x、wv/wo 恒等初始化】的设计原因(尺度自洽,实测关键):
      视觉塔 soft token 幅值极大(后续才被 embedder.encode_vision 归一),
      若 v 也走 LN(单位尺度),初始输出 ~O(0.02) 而回归靶 ~O(10²⁺),
      热身要先花几个量级步数"长尺度";v=原始 x + wv=wo=I → 初始输出
      = 注意力加权的源 token 均值(软选择语义),与靶同尺度、一步进入正题。
      q/k 路径保留 LN:注意力 logits 数值稳定。
    kind: query/fan_in/eye/ones/zeros(初始化语义见 resampler_init_leaf)。"""
    specs = [("tok_resampler_q", (total, D), "query")]
    for i in range(layers):
        pre = f"tok_resampler_l{i}_"
        specs += [(pre + "lnq_s", (D,), "ones"), (pre + "lnq_b", (D,), "zeros"),
                  (pre + "lnkv_s", (D,), "ones"), (pre + "lnkv_b", (D,), "zeros"),
                  (pre + "wq", (D, D), "fan_in"), (pre + "wk", (D, D), "fan_in"),
                  (pre + "wv", (D, D), "eye"), (pre + "wo", (D, D), "eye")]
        if ffn:
            H = D * ffn_mult
            specs += [(pre + "ln2_s", (D,), "ones"), (pre + "ln2_b", (D,), "zeros"),
                      (pre + "w1", (D, H), "fan_in"), (pre + "b1", (H,), "zeros"),
                      (pre + "w2", (H, D), "fan_in"), (pre + "b2", (D,), "zeros")]
    return specs


def resampler_init_leaf(name, shape, rng):
    """重采样器单叶宿主侧初始化(np.RandomState;train_sft 走 eval_shape 不
    物化参数,真实初值由本函数产生 —— 与 tok_scorer 的 tok0 初始化同机制)。
    规则按叶名后缀(与 _resampler_specs 的 kind 一一对应):
      LN scale(*_s)=1、各类 bias=0、query=N(0,0.02)、wq/wk/w1/w2=
      N(0,1/√fan_in)、wv/wo=单位阵(初始输出=注意力加权源 token 均值,
      与靶同尺度 —— 理由见 _resampler_specs 注释)。"""
    base = name.split("/")[-1]
    if base.endswith("_s"):
        return np.ones(shape, np.float32)
    if base.endswith(("_b", "b1", "b2")):
        return np.zeros(shape, np.float32)
    if base == "tok_resampler_q":
        return rng.normal(0, 0.02, shape).astype(np.float32)
    if base.endswith(("_wv", "_wo")):
        return np.eye(shape[0], dtype=np.float32)
    return rng.normal(0, 1.0 / shape[0] ** 0.5, shape).astype(np.float32)


def _resample_soft_tokens(t, prm, heads):
    """学习式重采样器前向(纯函数,可 CPU 单测)。
    t: [B, n, C, D] 全帧 soft token;prm: {叶名: 数组}(_resampler_specs 全集,
    层数/有无 FFN 从键名推断 → 前向与参数清单永远自洽);heads: 注意力头数
    (需整除 D)。返回 [B, total, D],total = prm["tok_resampler_q"].shape[0]。
    与硬选择的关键差异:输出 token 不是源 token 的子集,而是 softmax 注意力
    加权和 —— 全程可导、无信息硬丢弃;query 経训练可专业化(如"门口区域"、
    "运动前景"等抽象槽位)。"""
    import jax.numpy as jnp
    from jax import nn as jnn
    B, n, C, D = t.shape
    x = jnp.reshape(t, (B, n * C, D))                    # 源序列 [B, N, D]
    q = jnp.broadcast_to(prm["tok_resampler_q"][None],
                         (B,) + prm["tok_resampler_q"].shape)
    total = q.shape[1]
    assert D % heads == 0, f"RESAMPLER_HEADS={heads} 需整除 D={D}"
    dh = D // heads

    def _ln(v, s, b):                                    # LayerNorm(数值稳)
        m = jnp.mean(v, axis=-1, keepdims=True)
        var = jnp.var(v, axis=-1, keepdims=True)
        return (v - m) / jnp.sqrt(var + 1e-6) * s + b

    def _split(v):                                       # [B,L,D]→[B,h,L,dh]
        return jnp.transpose(
            jnp.reshape(v, (B, v.shape[1], heads, dh)), (0, 2, 1, 3))

    li = 0
    while f"tok_resampler_l{li}_wq" in prm:
        def p(s_, _i=li):
            return prm[f"tok_resampler_l{_i}_{s_}"]
        qn = _ln(q, p("lnq_s"), p("lnq_b"))
        kvn = _ln(x, p("lnkv_s"), p("lnkv_b"))
        # v 读原始 x(非 LN):初始 wv=wo=I 时输出=注意力加权源 token 均值,
        # 与视觉塔 soft token 同尺度(软选择语义;理由见 _resampler_specs)
        qh, kh, vh = _split(qn @ p("wq")), _split(kvn @ p("wk")), \
            _split(x @ p("wv"))
        att = jnn.softmax(
            jnp.einsum("bhqd,bhkd->bhqk", qh, kh) / (dh ** 0.5), axis=-1)
        o = jnp.einsum("bhqk,bhkd->bhqd", att, vh)       # [B,h,total,dh]
        o = jnp.reshape(jnp.transpose(o, (0, 2, 1, 3)), (B, total, D))
        q = q + o @ p("wo")                              # 残差
        if f"tok_resampler_l{li}_w1" in prm:             # 可选 FFN(残差)
            h = _ln(q, p("ln2_s"), p("ln2_b"))
            q = q + jnn.gelu(h @ p("w1") + p("b1")) @ p("w2") + p("b2")
        li += 1
    return q


def install_token_select():
    """池化后 soft token 压缩补丁(env SELECT_TOKENS_K>0 时启用;
    TOKEN_COMPRESS_MODE=topk|hyb)。位置:vision_encoder 输出(每帧
    MAX_SOFT_TOKENS 个真 soft token)→ _compress_soft_tokens →
    embedder.encode_vision。固定预算=静态形状(NPU 友好)。对任意 B
    生效(infer 的 B=1 与 train 的批路径都覆盖),替代 batched 补丁。"""
    import os
    import jax.numpy as jnp
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    if getattr(g4_tr, "_TOKEN_SELECT_PATCHED", False):
        return
    k = int(os.environ.get("SELECT_TOKENS_K", "0"))
    cnt_full = int(os.environ.get("MAX_SOFT_TOKENS", "64"))
    mode = os.environ.get("TOKEN_COMPRESS_MODE", "topk")
    seg_mode = mode == "dynseg"
    tome_mode = mode == "tome"
    seg_total = int(os.environ.get("DYNSEG_TOTAL", "512"))
    tome_total = int(os.environ.get("TOME_TOTAL", "512"))
    # 学习式重采样器(env TOKEN_RESAMPLER=1,默认关;设计见上方章节注释)
    rsp_mode = os.environ.get("TOKEN_RESAMPLER", "0") == "1"
    rsp_total = int(os.environ.get("RESAMPLER_TOKENS", "512"))
    rsp_layers = int(os.environ.get("RESAMPLER_LAYERS", "1"))
    rsp_heads = int(os.environ.get("RESAMPLER_HEADS", "8"))
    rsp_ffn = os.environ.get("RESAMPLER_FFN", "1") == "1"
    rsp_ffn_mult = int(os.environ.get("RESAMPLER_FFN_MULT", "2"))
    assert not (rsp_mode and (seg_mode or tome_mode)), \
        "TOKEN_RESAMPLER=1 与 dynseg/tome 互斥"
    assert 1 <= rsp_layers <= 2, "RESAMPLER_LAYERS 支持 1~2 层"
    # dynseg/tome/resampler 不依赖 SELECT_TOKENS_K
    assert seg_mode or tome_mode or rsp_mode or k > 0
    learn = os.environ.get("TOKEN_LEARN_SCORE", "0") == "1"
    lrank = int(os.environ.get("TOKEN_LEARN_RANK", "16"))

    # 可学习打分头参数注册(env TOKEN_LEARN_SCORE=1,默认关)。Transformer 是
    # setup 式模块(非 @compact)→ self.param 只能在 setup 内建;故补 setup
    # 包装,在原 setup 之后注册 tok_scorer_A[D,r]/tok_scorer_b[r,1](D=视觉塔
    # d_model)。b 暖启为 0(见 _compress_soft_tokens 的头:b=0 → 附加分≡0 →
    # 选择逐位不变)。关闭时不建任何参数 → params pytree 与现状逐位一致。
    if learn and not getattr(g4_tr, "_TOK_SCORER_PATCHED", False):
        import jax
        _orig_setup = g4_tr.Transformer.setup

        def _setup_learn(self):
            _orig_setup(self)
            ve = self.config.vision_encoder
            if ve is None:
                return
            D = ve.d_model
            # gm 的 _dtype_params 包装以 (key, shape, dtype) 三参调用 init_fn
            self.tok_scorer_A = self.param(
                "tok_scorer_A",
                lambda key, shape, dtype=jnp.float32: (
                    jax.random.normal(key, shape, jnp.float32)
                    * (1.0 / shape[0] ** 0.5)),
                (D, lrank))
            self.tok_scorer_b = self.param(
                "tok_scorer_b",
                lambda key, shape, dtype=jnp.float32: jnp.zeros(
                    shape, jnp.float32),
                (lrank, 1))

        g4_tr.Transformer.setup = _setup_learn
        g4_tr._TOK_SCORER_PATCHED = True

    # 重采样器参数注册(env TOKEN_RESAMPLER=1,默认关)。与 tok_scorer 同机制:
    # Transformer 是 setup 式模块(非 @compact)→ self.param 只能在 setup 内建;
    # 补 setup 包装,在原 setup 之后按 _resampler_specs 注册全部叶(D=视觉塔
    # d_model)。参数经 self.param 进 params pytree → 训练侧可训、随产物保存;
    # 前向在 _selected(model.apply 内部)经 self.tok_resampler 直接可达。
    # 关闭时不建任何参数 → params pytree 与现状逐位一致。
    if rsp_mode and not getattr(g4_tr, "_TOK_RESAMPLER_PATCHED", False):
        import jax
        _orig_setup_r = g4_tr.Transformer.setup

        def _mk_rsp_init(kind):
            # gm 的 _dtype_params 包装以 (key, shape, dtype) 三参调用 init_fn。
            # 注意:train_sft 只 eval_shape(不物化),真实初值由宿主侧
            # resampler_init_leaf 给出 —— 此处 init 仅为真跑 model.init 的
            # 场景(如单测)保持同语义。
            def _init(key, shape, dtype=jnp.float32):
                if kind == "ones":
                    return jnp.ones(shape, jnp.float32)
                if kind == "zeros":
                    return jnp.zeros(shape, jnp.float32)
                if kind == "eye":
                    return jnp.eye(shape[0], dtype=jnp.float32)
                if kind == "query":
                    return jax.random.normal(key, shape, jnp.float32) * 0.02
                return jax.random.normal(key, shape, jnp.float32) \
                    * (1.0 / shape[0] ** 0.5)
            return _init

        def _setup_rsp(self):
            _orig_setup_r(self)
            ve = self.config.vision_encoder
            if ve is None:
                return
            prm = {}
            for name, shape, kind in _resampler_specs(
                    ve.d_model, rsp_total, rsp_layers, rsp_ffn, rsp_ffn_mult):
                prm[name] = self.param(name, _mk_rsp_init(kind), shape)
            self.tok_resampler = prm

        g4_tr.Transformer.setup = _setup_rsp
        g4_tr._TOK_RESAMPLER_PATCHED = True

    def _selected(self, vision_input):
        patches = vision_input.patches                  # [B, n*p, d]
        counts = vision_input.soft_token_counts
        # dyn/tome/resampler 单块 counts 长度=1,真实帧数从 patch 总数反推
        # (每帧 9*mst patch)
        n = patches.shape[1] // (cnt_full * 9) \
            if (mode in ("dyn", "tome") or rsp_mode) else len(counts)
        B = patches.shape[0]
        p, d = patches.shape[1] // n, patches.shape[2]
        pa = jnp.reshape(patches, (B * n, p, d))
        px = jnp.reshape(vision_input.positions_xy, (B * n, p, 2))
        emb, _mask = self.vision_encoder(pa, px)[0]
        t = jnp.reshape(emb[:, :cnt_full, :], (B, n, cnt_full, -1))
        # 满分辨率 KD 老师:跳过一切压缩,直接返回全部 n*cnt_full 个 soft token
        # (16帧×64=1024)。老师看满 1024,学生仍走下方压缩到 512 —— KD 上界由此
        # 解封。vision_encoder 前向与学生完全相同(逐帧全跑),仅省去压缩步;返回
        # 的 1024 个 embedding 填满 layout 的 1024 视觉槽,T=1359 与学生一致。
        if getattr(vision_input, "full_res", False):
            sel_full = jnp.reshape(t, (B, n * cnt_full, t.shape[-1]))
            return self.embedder.encode_vision(sel_full[:, None])[:, 0]
        # 可学习打分头(默认关):取 setup 注册的参数;training 由 env
        # TOKEN_LEARN_TRAIN 控(train_sft 训练态置 1 开 STE 软门;推理/导出
        # 留空=纯硬选、逐位等价现状)。STE 前向不变故 eval 亦零影响。
        head, training = None, False
        if learn:
            A_h = getattr(self, "tok_scorer_A", None)
            if A_h is not None:
                head = (A_h.astype(t.dtype),
                        self.tok_scorer_b.astype(t.dtype))
                training = os.environ.get("TOKEN_LEARN_TRAIN", "0") == "1"
        ext = getattr(vision_input, "ext_score", None)  # 归因显著性(数据侧,可选)
        if rsp_mode:
            # 学习式重采样器:替换全部打分/硬选逻辑(score/head/ext 均不用)。
            # 参数可达性:_selected 作为 Transformer 的方法在 model.apply 内
            # 执行,self 是 flax 绑定模块 —— patched setup 里 self.param 注册
            # 的 self.tok_resampler 在此直接可读(tok_scorer 同款通道,已在
            # learnhead 训练/推理全链路验证过)。
            prm = getattr(self, "tok_resampler", None)
            assert prm is not None, ("TOKEN_RESAMPLER=1 但 setup 补丁未注册 "
                                     "tok_resampler_*(补丁在模型构造后才装?)")
            prm = {kk: vv.astype(t.dtype) for kk, vv in prm.items()}
            sel = _resample_soft_tokens(t, prm, rsp_heads)
        elif seg_mode:
            # seg_counts 逐样本逐帧预算 [B,n](数据侧,经 DynVisionInput pytree)
            seg = vision_input.seg_counts
            sel = _compress_soft_tokens(t, 0, "dynseg", seg=seg,
                                        total=seg_total, head=head,
                                        training=training, ext_score=ext)
        elif tome_mode:
            sel = _compress_soft_tokens(t, 0, "tome", total=tome_total,
                                        head=head, training=training,
                                        ext_score=ext)
        else:
            sel = _compress_soft_tokens(t, k, mode, head=head,
                                        training=training, ext_score=ext)
        return self.embedder.encode_vision(sel[:, None])[:, 0]

    g4_tr.Transformer._encode_vision = _selected
    g4_tr._TOKEN_SELECT_PATCHED = True
    _tag = ("TOTAL=" + str(seg_total) if seg_mode
            else "TOTAL=" + str(tome_total) if tome_mode
            else f"RSP={rsp_total} L={rsp_layers} H={rsp_heads} "
                 f"FFN={int(rsp_ffn)}" if rsp_mode
            else "K=" + str(k))
    _m = "resampler" if rsp_mode else mode
    print(f"[token-select] 池化后压缩已启用 mode={_m} {_tag}/{cnt_full}")


def install_batched_encode_vision():
    """gm 官方 _encode_vision 写死 B=1(reshape 忽略 batch 维);merge 侧
    vmap 天然支持 [B,T,D]。此补丁 B=1 走原路径,B>1 批量展开编码后折回。
    语义经 poc/05 等价测试钉死: batch=2 ≡ 2×bs1,max|Δ|<1e-4。"""
    import jax.numpy as jnp
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    if getattr(g4_tr, "_BATCH_EV_PATCHED", False):
        return
    _orig = g4_tr.Transformer._encode_vision

    def _batched(self, vision_input):
        patches = vision_input.patches
        B = patches.shape[0]
        if B == 1:
            return _orig(self, vision_input)
        counts = vision_input.soft_token_counts
        n, cnt = len(counts), counts[0]
        assert all(c == cnt for c in counts), "非均匀 counts 需回退 bs=1"
        p, d = patches.shape[1] // n, patches.shape[2]
        pa = jnp.reshape(patches, (B * n, p, d))
        px = jnp.reshape(vision_input.positions_xy, (B * n, p, 2))
        emb, _mask = self.vision_encoder(pa, px)[0]
        toks = emb[:, :cnt, :]                # 均匀正方形帧无 pad → 前 cnt 即真
        toks = jnp.reshape(toks, (B, n * cnt, toks.shape[-1]))
        return self.embedder.encode_vision(toks[:, None])[:, 0]

    g4_tr.Transformer._encode_vision = _batched
    g4_tr._BATCH_EV_PATCHED = True
