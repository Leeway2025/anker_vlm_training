"""JAX SFT 训练循环 v2(多芯数据并行 + 视觉 LoRA + projector 全参)。

  /dev/shm/venv_jax/bin/python jax_impl/train_sft.py \
      --labels /dev/shm/fakedata/labels.jsonl --layout /dev/shm/hf_layout.json \
      --steps 10 --accum 2 --dp 4 --train-vision --train-projector

v1→v2:
  - shard_map 数据并行: 每设备沿用已验证的 bs1 路径,梯度 pmean;
  - --train-vision: VisionBlock nn.remat(纯数组签名,免闭包技巧)
    + 视觉 LoRA 解冻(v1 消融: 无 remat 的视觉反向吃 ~32G);
  - --train-projector: embedder/mm_input_projection(+norm)全参训练
    (对齐 torch 路线: projector 不在 adapter 内、单独训练);
  - 梯度裁剪 1.0(对齐 torch max_grad_norm)+ 每 K 步验证集 loss。
v1 的七个坑及修复见 FINDINGS.md;本文件保留全部关键注释。
"""
import argparse
import contextlib
import dataclasses
import functools
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--wds-dir", default=None,
                    help="显式指定分片目录(覆盖 labels 内 meta.wds_dir)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--dp", type=int, default=0, help="数据并行设备数,0=全部")
    ap.add_argument("--per-device-bs", type=int, default=1)
    ap.add_argument("--prefetch-workers", type=int, default=8,
                    help="0=关闭预取(同步取数,调试用)")
    ap.add_argument("--rank-scheme", choices=["uniform", "prod", "map"],
                    default="uniform",
                    help="prod=生产方案: 差异化 rank 512/256 + rsLoRA α=2r;"
                         "map=变秩折叠产物续训(svd_truncate --rank-map,"
                         "秩查 --init-npz、前向 scale=1,修补轮用)")
    ap.add_argument("--lr", type=float, default=None,
                    help="缺省自动: prod=2e-5(v7 冷配方)/ uniform=1e-4")
    ap.add_argument("--proj-lr", type=float, default=5e-4)
    ap.add_argument("--vision-lr", type=float, default=2e-5,
                    help="视觉塔 LoRA 学习率(torch 生产: 2e-5)")
    ap.add_argument("--tok-lr", type=float, default=1e-3,
                    help="可学习打分头 lr(仅 env TOKEN_LEARN_SCORE=1 生效;"
                         "b 从 0 暖启,需较 LoRA 大的 lr 先让 b 离零)")
    ap.add_argument("--loraplus-ratio", type=float, default=None,
                    help="缺省自动: prod=1(v7 冷配方)/ uniform=16")
    ap.add_argument("--warmup", type=int, default=300,
                    help="warmup 步数(opt step;torch 生产: 300)")
    ap.add_argument("--lr-schedule", choices=["linear", "constant"],
                    default="linear",
                    help="linear=warmup+线性衰减到 0(torch Trainer 默认)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="torch 生产 wd=0;旧版隐含 optax 默认 1e-4")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle 与 val 切分种子")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--train-vision", action="store_true")
    ap.add_argument("--train-projector", action="store_true")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="连续 N 次 eval 无 val 改善即停(0=关;torch 同款"
                         "语义 patience=3;best 权重始终已落盘)")
    ap.add_argument("--val-n", type=int, default=8)
    ap.add_argument("--val-ids", default=None,
                    help="固定 val 卷子文件(每行一个 video_id;传入后 "
                         "--val-n 失效,所有实验在同一标尺上选 best)")
    ap.add_argument("--out", default="/dev/shm/out_jax_sft")
    ap.add_argument("--stage", choices=["a", "b"], default="b",
                    help="a=仅 projector 预热(LoRA 全冻结)")
    ap.add_argument("--sample-weights", help="hard-mining sw.json")
    ap.add_argument("--augment", action="store_true",
                    help="训练增强: 翻转/亮度/帧dropout(val 不增强;"
                         "时序翻转结构性禁止)")
    ap.add_argument("--cartography", action="store_true",
                    help="训练动力学日志: 每样本每次相遇的 CE →"
                         "<out>/cartography.jsonl(零额外前向;清洗夜标配,"
                         "错标 vs 真难例的第四证人)")
    ap.add_argument("--augment-v2", action="store_true",
                    help="增强包 v2: crop-zoom/对比度/轻遮挡(叠加在 v1 "
                         "之上;默认关,须在干净标签 100k 上 from-scratch "
                         "消融过门后方可进 1M 配方)")
    ap.add_argument("--augment-v3", action="store_true",
                    help="增强包 v3: 域定向(时序连续窗/灰度IR/低照度+噪声/"
                         "鱼眼warp),攻具体域偏移;叠加在 v1/v2 上,默认关,"
                         "同 v2 须 from-scratch 100k 消融过门后进配方")
    ap.add_argument("--aux-file", help="资产 A attributes.jsonl → 7 属性头")
    ap.add_argument("--aux-conf-threshold", type=float, default=0.5,
                    help="低于此置信度的标注整条屏蔽(torch 同款)")
    ap.add_argument("--aux-coef", type=float, default=0.3)
    ap.add_argument("--teacher-npz", default=None,
                    help="蒸馏老师产物(uniform 单一 rank,由 "
                         "svd_truncate_lora --pad-to-uniform 产出);"
                         "老师冻结前向 + KL 对齐,步耗时约 ×1.8。"
                         "仅支持 uniform 学生(小 rank 修复轮)")
    ap.add_argument("--teacher-npz2", default=None,
                    help="第二蒸馏老师(0814 合议KD:与老师1同 uniform rank,"
                         "两师 logits 在线平均后做 KL;步耗时再 +~35%%)")
    ap.add_argument("--distill-coef", type=float, default=0.5,
                    help="KL 蒸馏损失系数(硬标签 CE 保持全额)")
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--ks-head", action="store_true", help="KS 父类头(6 类)")
    ap.add_argument("--ks-coef", type=float, default=0.2)
    ap.add_argument("--cot-file", help="资产 C reasoning.jsonl → 隐式 CoT")
    ap.add_argument("--cot-ratio", type=float, default=0.6)
    ap.add_argument("--idw", type=float, default=0.0,
                    help="desc 身份词 token 权重(0=关;方案三用 3)")
    ap.add_argument("--rt-w", type=float, default=0.0,
                    help="RT 字母位 loss 权重(0=沿用 cls-w x4;④ RT位加权"
                         "实验用 8: RT决策只占总loss千分之几,单独提权逼"
                         "模型编码身份特征,SK/desc 绝对权重不变)")
    ap.add_argument("--cot-anneal", type=float, default=0.5,
                    help="最后该比例的步数切纯生产模式")
    ap.add_argument("--init-npz", help="从 train_params.npz 续训(或 import_hf 产物)")
    ap.add_argument("--remat-policy", choices=["full", "dots"],
                    default="full",
                    help="LLM 层重算策略: full=nothing_saveable(全重算,最省"
                         "显存,历史默认)/ dots=保留矩阵乘输出(反向少重算,"
                         "~+10-25% 吞吐,多吃显存)。数学恒等零配方风险,"
                         "OOM 风险由 speed_gate 实测把关")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help=">0: 每 N 个 opt step 落全量断点(参数+优化器+进度),"
                         "原子写、滚动保留 2 份;跨天长跑(1M)必开。"
                         "注意与 --init-npz 的区别: init-npz 只热启参数,"
                         "丢优化器动量/日程/数据位置")
    ap.add_argument("--resume", action="store_true",
                    help="从 <out>/ckpt_latest.npz 精确续跑(优化器动量、lr "
                         "日程、数据位置、best/patience 全恢复);断点不存在"
                         "时静默从头跑 —— 因此可无脑配 watchdog 外壳")
    ap.add_argument("--mu-dtype", choices=["bfloat16", "float32"],
                    default="bfloat16",
                    help="Adam 一阶动量精度(bf16 省 HBM ~2-4G,为 bs2/"
                         "松 remat 腾空间;LoRA 微调实证无损)")
    ap.add_argument("--profile-steps", type=int, default=0,
                    help=">0: 抓 opt step 10 起 N 步的 XLA trace 到 "
                         "<out>/tb_profile(TensorBoard 打开看 timeline)")
    ap.add_argument("--wandb", action="store_true",
                    help="上报 wandb(需 WANDB_API_KEY;指标/超参上云,"
                         "先过数据合规再开;离线用 WANDB_MODE=offline)")
    a = ap.parse_args()
    # v7 防过热: prod 的 rsLoRA scale(32/45)叠加热 lr/LoRA+ 会训死
    # 视觉→字母回路 —— prod 缺省自动用已验证冷配方,显式传参可覆盖
    # (覆盖越线时下方哨兵会警告)
    if a.lr is None:
        a.lr = 2e-5 if a.rank_scheme in ("prod", "map") else 1e-4
    if a.loraplus_ratio is None:
        a.loraplus_ratio = 1.0 if a.rank_scheme in ("prod", "map") else 16.0
    from jax_impl.logtee import tee_stdio
    tee_stdio(a.out)
    if a.stage == "a":
        a.train_projector = True     # stage a 语义: 只训 projector

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import flax.linen as nn
    from jax.sharding import Mesh, PartitionSpec as P
    from jax.experimental.shard_map import shard_map
    from gemma import gm
    from gemma import peft as gpeft
    from gemma.gm.nn.gemma4 import _modules as g4_modules
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from gemma.gm.nn.gemma4.vision import _transformer as gv_tr
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput

    from jax_impl.data import (SftDataset, make_vision_input,
                               install_batched_encode_vision,
                               make_dyn_vision_input, make_full_vision_input)
    _DYNSEG = os.environ.get("TOKEN_COMPRESS_MODE") == "dynseg"
    # tome:时空 ToMe 单块合并(默认关)。设备侧确定式合并,无逐样本元数据 →
    # 走原生 PreprocessedVisionInput(与非压缩路径同签名),仅 T=tome_T 变。
    _TOME = os.environ.get("TOKEN_COMPRESS_MODE") == "tome"
    # 可学习打分头(env TOKEN_LEARN_SCORE=1,默认关):训练态开 STE 软门,
    # 参数 tok_scorer_* 由 data.install_token_select 的 patched setup 用
    # self.param 注册进 params pytree,这里抽成独立可训子树 train["tok"]。
    _LEARN = os.environ.get("TOKEN_LEARN_SCORE", "0") == "1"
    if _LEARN:
        os.environ["TOKEN_LEARN_TRAIN"] = "1"     # 训练前向开 STE(推理留空)
        if a.teacher_npz or a.teacher_npz2:
            raise SystemExit(
                "TOKEN_LEARN_SCORE=1 暂不支持蒸馏: 老师 npz 无 tok_scorer_* "
                "参数,老师前向会缺参报错。请先无蒸馏训练打分头。")
    # 学习式重采样器(env TOKEN_RESAMPLER=1,默认关):参数 tok_resampler_*
    # 由 data.install_token_select 的 patched setup 用 self.param 注册进
    # params pytree,与打分头共用 train["tok"] 可训子树/优化器 "tok" 组/
    # npz "tok/<路径>" 保存通道。前向天然可导(softmax 注意力),无需 STE。
    _RSP = os.environ.get("TOKEN_RESAMPLER", "0") == "1"
    # int4 量化感知训练(env QAT_INT4=1,默认关):对可训 LoRA 叶做 int4 假量化
    # (逐 last-axis 通道对称,与部署 quantize_lora.py 逐位同口径),STE 直通梯度
    # → 权重学会适应 4bit 舍入,补回 PTINT4 的精度损耗。不改存储(仍 int4)。
    _QAT4 = os.environ.get("QAT_INT4", "0") == "1"
    if _RSP and (a.teacher_npz or a.teacher_npz2):
        raise SystemExit(
            "TOKEN_RESAMPLER=1 暂不支持蒸馏: 老师前向走同一 patched setup,"
            "老师 npz 无 tok_resampler_* 参数会缺参报错。请先无蒸馏训练。")

    devs = jax.devices()
    DP = a.dp or len(devs)
    BS = a.per_device_bs
    print(f"[env] devices={len(devs)} dp={DP} per_device_bs={BS} "
          f"global_micro={DP*BS}")
    if BS > 1:
        install_batched_encode_vision()
    if a.rank_scheme == "prod":
        from jax_impl.prod_lora import install_prod_lora
        install_prod_lora()      # 必须在模型构造前(patch 参数创建路径)
    elif a.rank_scheme == "map":
        # 变秩折叠续训: 秩表来自 --init-npz(必须带折叠标记)。
        # 产物保存时回写标记(下方 savez),否则混合秩会被误判 prod
        # → infer 二次缩放静默错。
        if not a.init_npz:
            raise SystemExit("--rank-scheme map 需要 --init-npz"
                             "(svd_truncate --rank-map 折叠产物)")
        from jax_impl.npz_io import detect_rank_scheme as _drs0
        _ms, _mmap = _drs0(np.load(a.init_npz))
        if _ms != "map":
            raise SystemExit(f"--rank-scheme map 但 init-npz 判定为 {_ms}"
                             "(无 __svd_scale_folded__ 标记),拒绝")
        from jax_impl.prod_lora import install_map_lora
        install_map_lora(_mmap)  # 必须在模型构造前

    # ---- 逐层重算(gm 无内置 remat;v1 坑 3/4)----
    if not getattr(g4_modules, "_REMAT_PATCHED", False):
        _orig_call = g4_modules.Block.__call__
        _POL = (jax.checkpoint_policies.nothing_saveable
                if a.remat_policy == "full" else
                jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        if a.remat_policy != "full":
            print(f"[remat] LLM 层策略={a.remat_policy}"
                  "(保留矩阵乘输出,反向少重算;盯 [hbm] 防 OOM)")

        def _make_core(skip_flag):
            def core(self, x, pos, cache, mask, pli, kvs):
                return _orig_call(self, x, pos, cache, mask, pli, kvs,
                                  skip_sliding_mask=skip_flag)
            return nn.remat(core, policy=_POL, prevent_cse=False)

        _core = {False: _make_core(False), True: _make_core(True)}

        def patched(self, x, segment_pos, cache, attn_mask,
                    per_layer_input=None, kv_shared_cache=None,
                    skip_sliding_mask=False):
            return _core[bool(skip_sliding_mask)](
                self, x, segment_pos, cache, attn_mask,
                per_layer_input, kv_shared_cache)

        g4_modules.Block.__call__ = patched
        g4_modules._REMAT_PATCHED = True

    if a.train_vision and not getattr(gv_tr, "_REMAT_PATCHED", False):
        # 视觉塔反向无 remat 时吃 ~32G(v1 消融);VisionBlock 纯数组签名
        gv_tr.VisionBlock = nn.remat(
            gv_tr.VisionBlock,
            policy=jax.checkpoint_policies.nothing_saveable)
        gv_tr._REMAT_PATCHED = True
        print("[remat] VisionBlock 已包装")

    # HF 语义对齐(v1 坑 7,勿删): gm 训练路径 remove_mm_logits 会压缩
    # 视觉位 logits,尾部垃圾盖住 label 窗口 → 恒等旁路
    g4_tr._token_utils.remove_mm_logits = \
        lambda logits, tokens, num_tokens_per_image: logits

    # ---- 模型(视频语义: output_length=64,Gate C 配方)----
    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    model = gm.nn.LoRA(rank=a.rank,
                       model=gm.nn.Gemma4_E2B(text_only=False, config=cfg64))

    from jax_impl.data import load_jsonl_map
    tok = gm.text.Gemma4Tokenizer()
    sw = json.load(open(a.sample_weights)) if a.sample_weights else None
    val_ids = (set(open(a.val_ids).read().split())
               if a.val_ids else None)
    # val_n 对齐 DP*BS: 旧版 val_n < DP*BS 时 eval 空转、val_loss 恒 0
    vn = 0
    if a.eval_every and a.val_n and not val_ids:
        g = DP * BS
        vn = ((a.val_n + g - 1) // g) * g
        if vn != a.val_n:
            print(f"[data] val_n {a.val_n} -> {vn}(对齐 DP*BS={g})")
    full = SftDataset(
        a.labels, a.layout, tok, wds_dir=a.wds_dir, sample_weights=sw,
        rt_weight=a.rt_w, id_weight=a.idw,
        reasoning=load_jsonl_map(a.cot_file) if a.cot_file else None,
        cot_ratio=a.cot_ratio,
        attributes=load_jsonl_map(a.aux_file) if a.aux_file else None,
        seed=a.seed, val_n=vn, aux_conf_threshold=a.aux_conf_threshold,
        augment=a.augment, augment_v2=a.augment_v2,
        augment_v3=a.augment_v3, val_ids=val_ids)
    train_idx, val_idx = full.train_idx, full.val_idx
    print(f"[data] train={len(train_idx)} val={len(val_idx)}"
          f"(按 camera 切分, seed={a.seed}, 先切后复制) "
          f"max_len={full.max_len}")

    # ---- lora 结构初始化(v1 坑 1: eval_shape 免物化)----
    ex = full[0]
    p0, x0, counts = make_vision_input([ex["frames"]])
    if _DYNSEG:
        # dynseg:seg_counts 走数据侧 pytree(逐帧变预算)。dummy 用首样本预算。
        dummy_pvi = make_dyn_vision_input(
            jnp.asarray(p0), jnp.asarray(x0), counts,
            jnp.asarray(ex["seg_counts"][None], jnp.int32))
    else:
        dummy_pvi = PreprocessedVisionInput(
            patches=jnp.asarray(p0), positions_xy=jnp.asarray(x0),
            soft_token_counts=counts)
    struct = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0),
        tokens=jnp.asarray(ex["tokens"][None]), images=dummy_pvi))
    _, lora_struct = gpeft.split_params(struct["params"])
    rng = np.random.RandomState(0)

    def _path_str(path):
        return "/".join(getattr(k, "key", str(k)) for k in path)

    # 可训集合 ≡ 可交付集合(torch prod targets / export_hf 映射):
    # gm.nn.LoRA 会给 per_layer_input_gate 等 PLE 模块也注入 LoRA,但
    # torch 生产不适配、HF 无导出映射 —— 训了也交付不了,还造成
    # JAX 评测与端侧成品的偏差,一律冻结(2026-07-20)
    EXPORTABLE = ("q_einsum", "kv_einsum", "attn_vec_einsum",
                  "gating_einsum", "/mlp/linear")

    def _is_trainable(pstr):
        if a.stage == "a":
            return False                     # stage a: LoRA 全冻结
        llm = pstr.startswith("layer_") or "/layer_" in pstr
        vis = (a.train_vision and "vision_encoder" in pstr
               and "stacked_layers" in pstr)  # entry 投影无 HF 注入点,不训
        if not (llm or vis):
            return False
        return any(k in pstr for k in EXPORTABLE)

    def init_leaf(path, leaf):
        pstr = _path_str(path)
        if pstr.endswith("/a") and _is_trainable(pstr):
            # peft gaussian 语义: std=1/r(r=叶末维)。旧值 0.02 偏大
            # 5-10×,叠加 rsLoRA scale 32/45 与 LoRA+ B×16 后有效步长
            # 超 torch 一个量级 → 长程视频→字母回路被打饱和,训练收敛
            # 到常数字母(2026-07-21 过拟合消融实锤: prod 常数/uniform
            # 100%;修复后 prod 亦 100%,见 FINDINGS v7)
            std = 1.0 / leaf.shape[-1]
            return jnp.asarray(rng.normal(0, std, leaf.shape), jnp.float32)
        return jnp.zeros(leaf.shape, jnp.float32)   # B 零 + 冻结项零
    lora0 = jax.tree_util.tree_map_with_path(init_leaf, lora_struct)

    # 可学习打分头子树: 从 struct 抽出 tok_scorer_* 叶(patched setup 注册,
    # 属 base 侧非-LoRA 参数,但运行时 base=ckpt 不含它们 → 单独成可训子树,
    # 前向时注入进 merged params)。b 暖启为 0(附加分≡0,选择逐位不变),
    # A 小随机(std=1/D)。init-npz 若无该叶则保留暖启。
    tok0 = {}
    if _LEARN or _RSP:
        from jax_impl.data import resampler_init_leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(
                struct["params"])[0]:
            ps = _path_str(path)
            if _LEARN and "tok_scorer" in ps:
                if ps.endswith("_b"):
                    tok0[ps] = jnp.zeros(leaf.shape, jnp.float32)   # 暖启=0
                else:
                    tok0[ps] = jnp.asarray(
                        rng.normal(0, 1.0 / leaf.shape[0], leaf.shape),
                        jnp.float32)
            elif _RSP and "tok_resampler" in ps:
                # 重采样器叶: 初始化语义单一事实源在 data.resampler_init_leaf
                # (LN=1/bias=0/query=0.02/矩阵=1/√fan_in/wo 小启)
                tok0[ps] = jnp.asarray(
                    resampler_init_leaf(ps, leaf.shape, rng), jnp.float32)
        if _LEARN and not any("tok_scorer" in k for k in tok0):
            raise SystemExit(
                "TOKEN_LEARN_SCORE=1 但 struct 中无 tok_scorer_* 参数 —— "
                "install_token_select 的 setup 补丁未生效?"
                "(确认 SELECT_TOKENS_K>0 或 dynseg/tome 已启用)")
        if _RSP and not any("tok_resampler" in k for k in tok0):
            raise SystemExit(
                "TOKEN_RESAMPLER=1 但 struct 中无 tok_resampler_* 参数 —— "
                "install_token_select 的 setup 补丁未生效?"
                "(确认 make_vision_input 已在模型构造前跑过一次)")
        print(f"[tok-score] 可训 token 头/重采样器: {sorted(tok0)} "
              f"(共 {sum(int(x.size) for x in tok0.values())} 参数)")

    base = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)

    # ---- projector 全参(从 base 抽出作为可训练树)----
    PROJ_KEYS = ("mm_input_projection",)   # checkpoint 实测仅此一项(与 torch projector tensors=1 一致)
    if a.train_projector:
        proj0 = {k: jax.tree.map(lambda x: jnp.asarray(x, jnp.float32),
                                 base["embedder"][k]) for k in PROJ_KEYS}
        print(f"[proj] 全参训练: {list(proj0)} "
              f"({sum(x.size for x in jax.tree.leaves(proj0))/1e6:.1f}M)")
    else:
        proj0 = {}
    base = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), base)

    # ---- 辅助头(7 属性)与 KS 父类头: 独立 fp32 小参数树 ----
    from data.taxonomy import AUX_VOCABS, AUX_HEAD_ORDER, KS_CLASSES
    D_MODEL = 1536
    aux0 = {}
    if a.aux_file:
        for h in AUX_HEAD_ORDER:
            n_cls = len(AUX_VOCABS[h])
            aux0[h] = {"w": jnp.asarray(
                rng.normal(0, 0.02, (D_MODEL, n_cls)), jnp.float32),
                "b": jnp.zeros((n_cls,), jnp.float32)}
    if a.ks_head:
        aux0["ks"] = {"w": jnp.asarray(
            rng.normal(0, 0.02, (D_MODEL, len(KS_CLASSES))), jnp.float32),
            "b": jnp.zeros((len(KS_CLASSES),), jnp.float32)}

    train0 = {"lora": lora0, "proj": proj0, "aux": aux0}
    if _LEARN or _RSP:
        train0["tok"] = tok0    # 打分头/重采样器(可训、随产物保存)
    if a.init_npz:                        # 续训: 覆盖同名叶(形状须一致)
        from jax_impl.npz_io import restore_train_tree
        z = np.load(a.init_npz)
        # ③号致命坑防护: npz 里的全零 LoRA a(如 stage a 产物,LoRA 全程
        # 冻结为零)不得覆盖新初始化 —— A=0∧B=0 梯度恒零,恢复即杀死适配器
        train0, st = restore_train_tree(
            train0, z, jnp, is_zero_skippable=_is_trainable)
        msg = f"[init-npz] 恢复 {st['hit']} 叶 from {a.init_npz}"
        if st["shape_skip"]:
            msg += (f";形状不符跳过 {st['shape_skip']} 叶"
                    f"(跨 rank 方案衔接时属预期,如 S1 uniform→S2 prod)")
        if st["zero_a_skip"]:
            msg += (f";⚠️ 全零 LoRA a 跳过 {st['zero_a_skip']} 叶"
                    f"(npz 来自未训 LoRA 的阶段,保留本次新初始化)")
        print(msg)
    # ---- 蒸馏老师(--teacher-npz): uniform 大 rank 冻结前向,KL 对齐 ----
    TEACH = {"lora": {}, "proj": {}}
    model_t = None
    if a.teacher_npz:
        if a.rank_scheme == "prod":
            raise SystemExit(
                "--teacher-npz 不支持 prod 学生: prod 前向带 rsLoRA scale "
                "(≠1),类级 patch 会给老师也套上 scale → 与 uniform 老师"
                "(scale=1)训练口径不一致。map(前向 scale=1,与 uniform "
                "老师同口径)已支持,见 prod_lora.teacher_rank 上下文覆盖")
        _map_teach = (a.rank_scheme == "map")   # map 老师需 rank 覆盖上下文
        from jax_impl.prod_lora import teacher_rank as _teacher_rank
        from jax_impl.npz_io import (detect_rank_scheme as _drs,
                                     load_lora_strict as _lls)
        zt = np.load(a.teacher_npz)
        t_scheme, t_ranks = _drs(zt)
        if t_scheme != "uniform":
            raise SystemExit(
                f"老师 npz 须 uniform 单一 rank(现 {t_ranks})—— 用 "
                "svd_truncate_lora.py --rank <R> --pad-to-uniform 重切")
        # ★ jax.eval_shape 的 trace 缓存按输入签名建键,但 LoRA rank 是经
        # flax ModuleInterceptor 注入的、不在 jit 缓存键里 —— 学生(rank a.rank)
        # 已在上面 eval_shape 过一次,老师若 rank 不同会命中学生的旧结构、
        # 静默拿到学生 rank(实测 r64 学生 + r512 老师 → 老师被算成 r64,
        # load_lora_strict 形状不符报错)。构建老师结构前清一次缓存即可,
        # 只影响 eval_shape 定结构;训练 apply 用具体参数不受影响。
        # map 学生: 上面 eval_shape 是变秩(逐路径),老师须清缓存并在
        # teacher_rank 上下文里按 uniform t_rank 建结构(否则命中变秩旧
        # trace 或被 rank_map 改写)。uniform 学生: 沿用原 rank≠判定。
        if _map_teach or t_ranks[0] != a.rank:
            jax.clear_caches()
        model_t = gm.nn.LoRA(rank=t_ranks[0],
                             model=gm.nn.Gemma4_E2B(text_only=False,
                                                    config=cfg64))
        _tctx = _teacher_rank(t_ranks[0]) if _map_teach \
            else contextlib.nullcontext()
        with _tctx:
            struct_t = jax.eval_shape(lambda: model_t.init(
                jax.random.PRNGKey(0),
                tokens=jnp.asarray(ex["tokens"][None]), images=dummy_pvi))
        lora_t = _lls(zt, gpeft.split_params(struct_t["params"])[1],
                      jnp, jnp.bfloat16)
        proj_t = {}
        for f in zt.files:                 # 老师 proj 同车(若产物携带)
            if not f.startswith("proj/"):
                continue
            segs = f.split("/")[1:]
            node = proj_t
            for s0 in segs[:-1]:
                node = node.setdefault(s0, {})
            node[segs[-1]] = jnp.asarray(zt[f], jnp.bfloat16)
        TEACH = {"lora": lora_t, "proj": proj_t}
        print(f"[distill] teacher rank={t_ranks[0]} coef={a.distill_coef} "
              f"temp={a.distill_temp} proj={'有' if proj_t else '无(用 base)'}")
        if a.teacher_npz2:
            # 合议KD(0814): 老师2 须与老师1同 uniform rank → 复用 struct_t,
            # 打包进同一 teach pytree("lora2"/"proj2"),train_step 签名零改动
            zt2 = np.load(a.teacher_npz2)
            t2_scheme, t2_ranks = _drs(zt2)
            if t2_scheme != "uniform" or t2_ranks[0] != t_ranks[0]:
                raise SystemExit(
                    f"老师2 须与老师1同 uniform rank({t_ranks[0]}),"
                    f"现 {t2_scheme}/{t2_ranks} —— pad-to-uniform 重切")
            lora_t2 = _lls(zt2, gpeft.split_params(struct_t["params"])[1],
                           jnp, jnp.bfloat16)
            proj_t2 = {}
            for f in zt2.files:
                if not f.startswith("proj/"):
                    continue
                segs = f.split("/")[1:]
                node = proj_t2
                for s0 in segs[:-1]:
                    node = node.setdefault(s0, {})
                node[segs[-1]] = jnp.asarray(zt2[f], jnp.bfloat16)
            TEACH["lora2"] = lora_t2
            TEACH["proj2"] = proj_t2
            print(f"[distill] 合议老师2 rank={t2_ranks[0]} "
                  f"proj={'有' if proj_t2 else '无'} —— 两师 logits 平均")

    # lr 日程: warmup + 线性衰减到 0(对齐 torch Trainer 默认;MultiSteps
    # 只在累积满时调用内层 update → 日程按 opt step 计数,与 torch 同拍)
    def mk_sched(peak):
        if a.lr_schedule == "constant":
            return peak
        return optax.join_schedules(
            [optax.linear_schedule(0.0, peak, max(a.warmup, 1)),
             optax.linear_schedule(peak, 0.0, max(a.steps - a.warmup, 1))],
            [max(a.warmup, 1)])

    # 分组对齐 torch 生产: vision LoRA 2e-5(旧版错用 1e-4,5×超速);
    # LoRA+ B 矩阵 lr×16;aux 头随 LLM-A 组;wd 显式 0(optax 默认 1e-4
    # 是 torch 没有的隐藏收缩力)
    def _group(p, _):
        k = _path_str(p)
        if k.startswith("tok/"):
            return "tok"
        if k.startswith("proj"):
            return "proj"
        vis = "vision_encoder" in k
        b = k.endswith("/b")
        if vis:
            return "vis_b" if b else "vis_a"
        return "llm_b" if b else "llm_a"
    GROUP_LR = {"proj": a.proj_lr,
                "llm_a": a.lr, "llm_b": a.lr * a.loraplus_ratio,
                "vis_a": a.vision_lr,
                "vis_b": a.vision_lr * a.loraplus_ratio,
                "tok": a.tok_lr}                       # 可学习打分头(不训时不用)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),         # 对齐 torch max_grad_norm
        optax.multi_transform(
            {g: optax.adamw(mk_sched(lr), weight_decay=a.weight_decay,
                            mu_dtype=a.mu_dtype)
             for g, lr in GROUP_LR.items()},
            param_labels=lambda tree: jax.tree_util.tree_map_with_path(
                _group, tree)))
    if a.rank_scheme == "prod" and a.lr * a.loraplus_ratio > 1e-4:
        print("⚠️ [optim] prod 方案 + 高 lr/LoRA+ 组合: rsLoRA scale(32/45)"
              "叠加后有效步长过热,实测会把视觉→字母回路训死"
              "(FINDINGS v7 过拟合消融)。已验证配方: --lr 2e-5 "
              "--loraplus-ratio 1")
    print(f"[optim] {a.lr_schedule} warmup={a.warmup} wd={a.weight_decay} "
          f"lr: llm_a={a.lr:g} llm_b={a.lr*a.loraplus_ratio:g} "
          f"vis_a={a.vision_lr:g} vis_b={a.vision_lr*a.loraplus_ratio:g} "
          f"proj={a.proj_lr:g}")
    optim = optax.MultiSteps(tx, every_k_schedule=a.accum)
    opt_state = optim.init(train0)

    # ---- 断点恢复(--resume): 参数+优化器+进度全量覆盖 ----
    # 与 --init-npz 的分工: init-npz=跨阶段热启(只参数,动量/日程归零),
    # resume=同一次跑的精确续接(中断点处一切如初)。两者同给时 resume 赢
    # (init-npz 先应用,随后被断点整树覆盖)。
    start_micro, _ck = 0, None
    if a.resume:
        from jax_impl.npz_io import load_ckpt
        r = load_ckpt(a.out, train0, opt_state, jnp)
        if r:
            train0, opt_state, _ck = r
            for k, want in (("seed", a.seed), ("accum", a.accum),
                            ("dp", DP), ("bs", BS), ("steps", a.steps),
                            ("cot_anneal", a.cot_anneal)):
                if _ck.get(k) != want:
                    raise SystemExit(
                        f"[resume] 断点 {k}={_ck.get(k)} 与本次 {want} 不一致"
                        " —— 拒绝续跑: 改配置=另一次实验,数据序/lr 日程都会"
                        "错位,请换 --out 从头跑")
            start_micro = int(_ck["micro_done"])
            print(f"[resume] 自 opt_step {_ck['opt_step']}/{a.steps} 续跑"
                  f"(micro {start_micro},best={_ck['best'][0]:.4f}"
                  f"@{_ck['best'][1]},since_best={_ck['since_best']})")
        else:
            print("[resume] 未发现可用断点,从头开始")

    ls = a.label_smoothing
    focal_gamma = float(os.environ.get("FOCAL_GAMMA", "0.0"))  # focal: down-weight easy (high-conf m) tokens
    pair_coef = float(os.environ.get("PAIR_MARGIN_COEF", "0.0"))  # pair-margin: 同父兄弟判别压力(SubKS 位)
    pair_margin = float(os.environ.get("PAIR_MARGIN", "3.0"))     # hinge 间隔
    # pair-margin 常量: SubKS 字母 token id(a..u, Gemma4 单 token)+ 同父兄弟掩码(KS_GROUP)
    _SK_IDS_NP = np.asarray(
        [236746, 236763, 236755, 236753, 236744, 236760, 236759, 236754,
         236747, 236804, 236767, 236752, 236757, 236749, 236748, 236758,
         236809, 236750, 236751, 236745, 236756], np.int32)      # a..u
    _SK_GROUPS = [list(range(0, 13)), [13, 14], [15, 16, 17],     # Normal / PropDmg / LifeThreat
                  [18], [19], [20]]                               # Loiter / VehAnom / UnauthEntry(单例无兄弟)
    _SIB_NP = np.zeros((21, 21), bool)
    for _g in _SK_GROUPS:
        for _i in _g:
            for _j in _g:
                if _i != _j:
                    _SIB_NP[_i, _j] = True
    T = (full.seg_T if _DYNSEG                           # dynseg 变长模板 T=seg_T
         else full.tome_T if _TOME                       # tome 单块模板 T=tome_T
         else len(full.template))
    mesh = Mesh(np.asarray(devs[:DP]), ("dp",))

    def loss_fn(train, base_p, teach, tokens, labels, weights, patches,
                pos_xy, aux_labels, ks_label):
        # v1 坑 5: fp32 参与会把激活链提升 fp32 → 前向统一 bf16;
        # 冻结 lora 叶 stop_gradient(v1 坑 6: 切断未训练子树反向)
        def _fake_int4(x):
            # 逐 last-axis 通道对称 int4,与部署 quantize_lora.py 同口径
            xf = x.astype(jnp.float32)
            axes = tuple(range(xf.ndim - 1))
            amax = jnp.max(jnp.abs(xf), axis=axes, keepdims=True)
            scale = jnp.where(amax > 0, amax / 7.0, 1.0)
            q = jnp.clip(jnp.round(xf / scale), -7, 7)
            return (q * scale).astype(jnp.bfloat16)

        def _lora_leaf(p, x):
            trn = _is_trainable(_path_str(p))
            xb = x.astype(jnp.bfloat16)
            if _QAT4:
                # 前向走 int4 假量化;可训叶 STE 直通梯度,冻结叶 stop_gradient
                fq = xb + jax.lax.stop_gradient(_fake_int4(xb) - xb)
                return fq if trn else jax.lax.stop_gradient(_fake_int4(xb))
            return xb if trn else jax.lax.stop_gradient(xb)

        lora_h = jax.tree_util.tree_map_with_path(_lora_leaf, train["lora"])
        base_h = base_p
        if a.train_projector:
            emb = dict(base_p["embedder"])
            for k in PROJ_KEYS:
                emb[k] = jax.tree.map(
                    lambda x: x.astype(jnp.bfloat16), train["proj"][k])
            base_h = dict(base_p)
            base_h["embedder"] = emb
        params = gpeft.merge_params(base_h, lora_h)
        if _LEARN or _RSP:
            # 打分头/重采样器参数注入 merged params 的注册路径(base=ckpt 不含,须补)。
            # 沿路径拷贝各层 dict(不改 base/train 的 pytree 结构),叶取自
            # train["tok"](可训、梯度回流),转 bf16 与前向一致。
            params = dict(params)
            for ps, leaf in train["tok"].items():
                segs = ps.split("/")
                node = params
                for s in segs[:-1]:
                    node[s] = dict(node[s])
                    node = node[s]
                node[segs[-1]] = leaf.astype(jnp.bfloat16)
        if _DYNSEG:
            # 逐帧预算从 tokens 尾部保留区取出(__getitem__ 编入,因果掩码之后、
            # label 屏蔽区,永不参与 loss)→ 作为数据侧 seg_counts 送达 _selected。
            # 无需改 shard_map 签名(tokens 本就是分片数据参数)→ 现有路径零改动。
            seg = jax.lax.dynamic_slice_in_dim(
                tokens, full.seg_off, full.seg_n, axis=1).astype(jnp.int32)
            pvi = make_dyn_vision_input(patches, pos_xy, counts, seg)
        else:
            pvi = PreprocessedVisionInput(
                patches=patches, positions_xy=pos_xy, soft_token_counts=counts)
        need_hidden = bool(train["aux"])
        out = model.apply({"params": params}, tokens=tokens, images=pvi,
                          return_hidden_states=need_hidden or None)
        logits = out.logits if hasattr(out, "logits") else out
        lg = logits[:, T - 1:-1].astype(jnp.float32)   # 尾窗(v1 坑 2)
        lb = labels[:, T:]
        wt = weights[:, T:]
        valid = (lb != -100).astype(jnp.float32)
        lse = jax.nn.logsumexp(lg, axis=-1)
        tgt = jnp.take_along_axis(
            lg, jnp.clip(lb, 0)[..., None], axis=-1)[..., 0]
        ce = (1 - ls) * (lse - tgt) + ls * (lse - lg.mean(-1))
        if focal_gamma > 0.0:                    # focal: down-weight easy (high-conf m) tokens
            pt = jnp.exp(jnp.clip(tgt - lse, -30.0, 0.0))
            ce = ((1.0 - pt) ** focal_gamma) * ce
        ce = jnp.where(valid > 0, ce, 0.0)      # PAD 行 NaN×0 防护
        loss = (ce * wt).sum() / jnp.clip((wt * valid).sum(), 1.0)
        # 训练动力学(cartography): 每样本加权 CE, 与 loss 同一批中间量,
        # 零额外前向。只含生产口径 CE(不混 KL/aux 头), 信号纯净 ——
        # 高波动低置信=疑似错标, 稳定高 loss=真难例(区分二者的第四证人)
        per_ex = (ce * wt).sum(-1) / jnp.clip((wt * valid).sum(-1), 1.0)

        # 满分辨率 KD 老师前向(env KD_TEACHER_FULLRES=1,默认关):老师看满
        # 1024 个 soft token(不压缩),学生仍压到 512 → 解封 KD 信息上界。仅替换
        # 老师前向的 images(pvi_t),学生前向/shard_map 签名/其余全部不变;full_res
        # 走 DynVisionInput 的 meta 侧,_selected 里 getattr 分支命中即跳过压缩。
        # student 侧 install_token_select 已把 _encode_vision patch 成 _selected,
        # 老师复用同一被 patch 的类,故 full_res 生效。默认关 → 不影响任何现有配方。
        pvi_t = pvi
        if os.environ.get("KD_TEACHER_FULLRES", "0") == "1":
            _mst_fr = int(os.environ.get("MAX_SOFT_TOKENS", "64"))
            _nfr = patches.shape[1] // (_mst_fr * 9)
            pvi_t = make_full_vision_input(patches, pos_xy, (_mst_fr,) * _nfr)

        if a.teacher_npz:                       # KL 蒸馏(老师冻结前向)
            lora_t_h = jax.tree.map(
                lambda x: jax.lax.stop_gradient(x.astype(jnp.bfloat16)),
                teach["lora"])
            base_t = base_p
            if teach["proj"]:
                emb_t = dict(base_p["embedder"])
                for k in teach["proj"]:
                    emb_t[k] = jax.tree.map(
                        lambda x: x.astype(jnp.bfloat16), teach["proj"][k])
                base_t = dict(base_p)
                base_t["embedder"] = emb_t
            params_t = gpeft.merge_params(base_t, lora_t_h)
            # map 学生: 老师前向须在 teacher_rank 上下文内 trace,eff_rank
            # 短路走 uniform 秩(否则被 rank_map 改写成变秩、shape 不符)。
            # 学生 model.apply(上方)在上下文外 → 照常查表。
            _t_ctx = (_teacher_rank(t_ranks[0]) if a.rank_scheme == "map"
                      else contextlib.nullcontext())
            with _t_ctx:
                out_t = model_t.apply({"params": params_t}, tokens=tokens,
                                      images=pvi_t)
            lg_t = jax.lax.stop_gradient(
                (out_t.logits if hasattr(out_t, "logits") else out_t)
                [:, T - 1:-1].astype(jnp.float32))
            if "lora2" in teach:                # 合议KD: 老师2 前向,logits 平均
                lora_t2_h = jax.tree.map(
                    lambda x: jax.lax.stop_gradient(x.astype(jnp.bfloat16)),
                    teach["lora2"])
                base_t2 = base_p
                if teach.get("proj2"):
                    emb_t2 = dict(base_p["embedder"])
                    for k in teach["proj2"]:
                        emb_t2[k] = jax.tree.map(
                            lambda x: x.astype(jnp.bfloat16),
                            teach["proj2"][k])
                    base_t2 = dict(base_p)
                    base_t2["embedder"] = emb_t2
                params_t2 = gpeft.merge_params(base_t2, lora_t2_h)
                # teacher_rank 是 generator 型上下文,不可复用 → 新建一个
                _t_ctx2 = (_teacher_rank(t_ranks[0])
                           if a.rank_scheme == "map"
                           else contextlib.nullcontext())
                with _t_ctx2:
                    out_t2 = model_t.apply({"params": params_t2},
                                           tokens=tokens, images=pvi_t)
                lg_t2 = jax.lax.stop_gradient(
                    (out_t2.logits if hasattr(out_t2, "logits") else out_t2)
                    [:, T - 1:-1].astype(jnp.float32))
                lg_t = 0.5 * (lg_t + lg_t2)
            tau = a.distill_temp
            p_t = jax.nn.softmax(lg_t / tau)
            kl = (p_t * (jax.nn.log_softmax(lg_t / tau)
                         - jax.nn.log_softmax(lg / tau))).sum(-1)
            kl = jnp.where(valid > 0, kl, 0.0)
            dloss = (kl * wt).sum() / jnp.clip((wt * valid).sum(), 1.0)
            loss = loss + a.distill_coef * (tau * tau) * dloss

        if need_hidden:
            hs = out.hidden_states
            hs = hs[-1] if isinstance(hs, (tuple, list)) else hs
            # 视觉位池化(哨兵 -2 位置): 属性/KS 都是视觉判断
            vmask = (tokens == -2).astype(jnp.float32)[..., None]
            pooled = ((hs.astype(jnp.float32) * vmask).sum(1)
                      / jnp.clip(vmask.sum(1), 1.0))            # [B, D]
            def head_ce(w, b, y):
                lgt = pooled @ w + b
                lp = jax.nn.log_softmax(lgt)
                ok = (y >= 0).astype(jnp.float32)
                pick = jnp.take_along_axis(
                    lp, jnp.clip(y, 0)[:, None], axis=-1)[:, 0]
                return -(pick * ok).sum() / jnp.clip(ok.sum(), 1.0)
            aux_l = 0.0
            n_heads = 0
            from data.taxonomy import AUX_HEAD_ORDER as _AHO
            for j, h in enumerate(_AHO):
                if h in train["aux"]:
                    aux_l += head_ce(train["aux"][h]["w"],
                                     train["aux"][h]["b"], aux_labels[:, j])
                    n_heads += 1
            if n_heads:
                loss = loss + a.aux_coef * aux_l / n_heads
            if "ks" in train["aux"]:
                loss = loss + a.ks_coef * head_ce(
                    train["aux"]["ks"]["w"], train["aux"]["ks"]["b"], ks_label)

        if pair_coef > 0.0:                     # pair-margin: 同父兄弟判别压力(SubKS 位)
            _SK = jnp.asarray(_SK_IDS_NP)                        # [21]
            _SIB = jnp.asarray(_SIB_NP)                          # [21,21] 同父兄弟(排除自身)
            _rows = jnp.arange(lg.shape[0])
            has_w = wt > 0.0                                     # think 权重恰为 0.0 → 排除;首个带权=RT 位
            rt_pos = jnp.argmax(has_w.astype(jnp.int32), axis=1)
            sk_pos = jnp.minimum(rt_pos + 2, lg.shape[1] - 1)    # SubKS 位(CoT 变长安全)
            sk_let = lg[_rows, sk_pos][:, _SK]                   # [B,21] 字母 logit
            true_id = lb[_rows, sk_pos]                          # [B] 金标 SubKS token id
            match = (true_id[:, None] == _SK[None, :])           # [B,21]
            true_idx = jnp.argmax(match.astype(jnp.int32), axis=1)
            true_logit = jnp.take_along_axis(
                sk_let, true_idx[:, None], axis=1)[:, 0]         # [B]
            sib = _SIB[true_idx].astype(jnp.float32)             # [B,21]
            hinge = jnp.clip(
                pair_margin - (true_logit[:, None] - sk_let), 0.0) * sib
            pair = hinge.sum(1) / jnp.clip(sib.sum(1), 1.0)      # [B] 兄弟均值
            row_ok = (has_w.any(1) & match.any(1)).astype(jnp.float32)
            pair_loss = (pair * row_ok).sum() / jnp.clip(row_ok.sum(), 1.0)
            loss = loss + pair_coef * pair_loss
        return loss, per_ex

    def grad_local(train, base_p, teach, tokens, labels, weights, patches,
                   pos_xy, aux_labels, ks_label):
        (loss, per_ex), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            train, base_p, teach, tokens, labels, weights, patches, pos_xy,
            aux_labels, ks_label)
        return (jax.lax.pmean(loss, "dp"), per_ex,
                jax.tree.map(lambda g: jax.lax.pmean(g, "dp"), grads))

    grad_sharded = shard_map(
        grad_local, mesh=mesh,
        in_specs=(P(), P(), P(), P("dp"), P("dp"), P("dp"), P("dp"),
                  P("dp"), P("dp"), P("dp")),
        out_specs=(P(), P("dp"), P()), check_rep=False)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def train_step(train, opt_state, base_p, teach, tokens, labels, weights,
                   patches, pos_xy, aux_labels, ks_label):
        loss, per_ex, grads = grad_sharded(train, base_p, teach, tokens,
                                           labels, weights, patches, pos_xy,
                                           aux_labels, ks_label)
        updates, opt_state = optim.update(grads, opt_state, train)
        train = optax.apply_updates(train, updates)
        return train, opt_state, loss, per_ex

    eval_local = shard_map(
        lambda tr, bp, tc, t, l, w, p, x, al, kl: jax.lax.pmean(
            loss_fn(tr, bp, tc, t, l, w, p, x, al, kl)[0], "dp"),
        mesh=mesh,
        in_specs=(P(), P(), P(), P("dp"), P("dp"), P("dp"), P("dp"),
                  P("dp"), P("dp"), P("dp")),
        out_specs=P(), check_rep=False)
    eval_loss_j = jax.jit(eval_local)

    def collect(idxs):
        exs = [full[i] for i in idxs]
        pt_all, px_all, _ = make_vision_input([e["frames"] for e in exs])
        pt, px = list(pt_all), list(px_all)
        return (jnp.asarray(np.stack([e["tokens"] for e in exs])),
                jnp.asarray(np.stack([e["labels"] for e in exs])),
                jnp.asarray(np.stack([e["weights"] for e in exs])),
                jnp.asarray(np.stack(pt)), jnp.asarray(np.stack(px)),
                jnp.asarray(np.stack([e["aux_labels"] for e in exs])),
                jnp.asarray(np.stack([e["ks_label"] for e in exs])))

    train = train0
    os.makedirs(a.out, exist_ok=True)
    _mf = open(os.path.join(a.out, "metrics.jsonl"), "a", buffering=1)
    # 训练动力学日志(--cartography): (video_id, micro, per-sample CE) →
    # <out>/cartography.jsonl。跑完用 均值(难度)×方差(波动) 二维图分桶:
    # 高波动低置信=疑似错标 / 稳定高 loss=真难例(Dataset Cartography/AUM)
    _carto_f = (open(os.path.join(a.out, "cartography.jsonl"), "a")
                if a.cartography else None)
    _carto_buf = []
    try:                                    # TB 可选(镜像无包则静默降级)
        from tensorboardX import SummaryWriter
        _tb = SummaryWriter(os.path.join(a.out, "tb"))
    except Exception:  # noqa: BLE001
        _tb = None
        print("[metrics] tensorboardX 不可用,仅写 metrics.jsonl")
    _wb = None
    if a.wandb:
        try:
            import wandb as _wb
            _wb.init(project="anker-vlm",
                     name=os.path.basename(a.out.rstrip("/")),
                     config=vars(a))
        except Exception as e:  # noqa: BLE001
            _wb = None
            print(f"[metrics] wandb 初始化失败,忽略: {e}")

    def _log_scalar(step, **kv):
        _mf.write(json.dumps({"step": step, **kv}) + "\n")
        if _tb:
            for k, v in kv.items():
                _tb.add_scalar(k, v, step)
        if _wb:
            _wb.log(kv, step=step)
    hist, best = [], (1e9, -1)
    if _ck:                                  # 进度状态随断点一并恢复
        hist = [float(x) for x in _ck["hist"]]
        best = (float(_ck["best"][0]), int(_ck["best"][1]))
        main._since_best = int(_ck["since_best"])
    cursor = start_micro * DP * BS
    t0 = time.time()
    total_micro = a.steps * a.accum
    # 每 epoch 用固定 seed 重洗(旧版严格顺序循环: 同类样本成段 →
    # batch 内梯度强相关;hard-mining 副本相邻 → 等效单步 lr×n)。
    # 全部置换预生成 → 线程安全、prefetch/同步路径逐条一致、可复现
    ep_len = len(train_idx)
    _rs = np.random.RandomState(a.seed)
    _perms = [_rs.permutation(ep_len)
              for _ in range(total_micro * DP * BS // ep_len + 2)]
    train_np = np.asarray(train_idx)

    def draw(k):
        e, i = divmod(k, ep_len)
        return int(train_np[_perms[e][i]])
    switch_at = int(total_micro * (1 - a.cot_anneal)) if a.cot_file else -1
    if switch_at >= 0 and start_micro >= switch_at:
        full.set_anneal(True)   # resume 落在退火段: 循环内的 == 触发点已过,
        print(f"[anneal] resume 于退火段(micro {start_micro} ≥ "
              f"{switch_at}),直接以纯生产模式续跑")
    pf = None
    if a.prefetch_workers > 0:
        from jax_impl.prefetch import BatchPrefetcher
        pf = BatchPrefetcher(full, draw,
                             DP * BS, workers=a.prefetch_workers)
        if start_micro:         # 丢掉构造时从 0 预取的批,跳到断点位置
            pf.flush(restart_at=start_micro * DP * BS)
        print(f"[prefetch] workers={a.prefetch_workers} depth=2 "
              f"(每 epoch 重洗, seed={a.seed}, start={start_micro * DP * BS})")
    for micro in range(start_micro, total_micro):
        if switch_at >= 0 and micro == switch_at and not full.anneal:
            full.set_anneal(True)
            if pf:                      # 清掉队列里旧模式 batch,边界零滞后
                pf.flush(restart_at=micro * DP * BS)
            print(f"[anneal] micro {micro}: 切换纯生产模式", flush=True)
        if pf:
            t_, l_, w_, p_, x_, al_, kl_, _ = pf.next()
            batch = tuple(jnp.asarray(v)
                          for v in (t_, l_, w_, p_, x_, al_, kl_))
        else:
            idxs = [draw(cursor + j) for j in range(DP * BS)]
            cursor += DP * BS
            batch = collect(idxs)
        train, opt_state, loss, per_ex = train_step(train, opt_state, base,
                                                    TEACH, *batch)
        if _carto_f is not None:
            # video_id 由 draw(纯函数) 宿主侧反推, 预取器零改动;
            # per_ex 顺序=batch 顺序=draw 顺序(shard_map P("dp") 按首轴切)
            _carto_buf.append((micro, [float(x) for x in np.asarray(per_ex)]))
            if len(_carto_buf) >= 64:
                for m_, ls_ in _carto_buf:
                    for j_, v_ in enumerate(ls_):
                        _carto_f.write(json.dumps({
                            "video_id": full.recs[draw(m_ * DP * BS + j_)]
                                            ["video_id"],
                            "micro": m_, "loss": round(v_, 4)}) + "\n")
                _carto_buf.clear()
        if micro == start_micro:
            print(f"[compile+step0] {time.time()-t0:.0f}s", flush=True)
            t0 = time.time()
        if (micro + 1) % a.accum == 0:
            opt_step = (micro + 1) // a.accum
            l = float(loss)              # 强制同步 → 边际耗时是真实的
            hist.append(l)
            now = time.time()
            dt = (now - getattr(main, "_tprev", t0)) / a.accum
            main._tprev = now
            print(f"[sft] opt_step {opt_step}/{a.steps} loss={l:.4f} "
                  f"marginal_micro_s={dt:.3f} "
                  f"samples/s={DP*BS/max(dt,1e-9):.1f}", flush=True)
            _log_scalar(opt_step, loss=l,
                        samples_per_s=DP * BS / max(dt, 1e-9))
            if a.profile_steps and opt_step == 10:
                jax.profiler.start_trace(os.path.join(a.out, "tb_profile"))
            if a.profile_steps and opt_step == 10 + a.profile_steps:
                jax.profiler.stop_trace()
                print(f"[profile] trace 已存 {a.out}/tb_profile", flush=True)
            if a.eval_every and opt_step % a.eval_every == 0:
                vl = []
                for k in range(0, len(val_idx) - DP * BS + 1, DP * BS):
                    vb = collect(val_idx[k:k + DP * BS])
                    vl.append(float(eval_loss_j(train, base, TEACH, *vb)))
                v = sum(vl) / max(len(vl), 1)
                tag = ""
                if vl and v < best[0]:
                    best = (v, opt_step); tag = " *best"
                    # best 即时落盘 —— 旧版只记 meta,交付的永远是最后
                    # 一步(过拟合了也照存)
                    bf = jax.tree_util.tree_flatten_with_path(train)[0]
                    _pl = {_path_str(p): np.asarray(x) for p, x in bf}
                    if a.rank_scheme == "map":   # 折叠标记随产物走(防误判 prod)
                        _pl["__svd_scale_folded__"] = np.array(1)
                    np.savez(os.path.join(a.out, "train_params_best.npz"),
                             **_pl)
                print(f"[eval] opt_step {opt_step} val_loss={v:.4f}{tag}",
                      flush=True)
                _log_scalar(opt_step, val_loss=v)
                if tag:
                    main._since_best = 0
                else:
                    main._since_best = getattr(main, "_since_best", 0) + 1
                    if (a.early_stop_patience
                            and main._since_best >= a.early_stop_patience):
                        print(f"[early-stop] 连续 {main._since_best} 次 eval "
                              f"无改善(best={best[0]:.4f}@{best[1]}),提前结束",
                              flush=True)
                        break
            if a.ckpt_every and opt_step % a.ckpt_every == 0:
                from jax_impl.npz_io import save_ckpt
                dt_ck = save_ckpt(a.out, train, opt_state, {
                    "micro_done": micro + 1, "opt_step": opt_step,
                    "best": [float(best[0]), int(best[1])],
                    "since_best": int(getattr(main, "_since_best", 0)),
                    "hist": [float(x) for x in hist],
                    "seed": a.seed, "accum": a.accum, "dp": DP, "bs": BS,
                    "steps": a.steps, "cot_anneal": a.cot_anneal})
                print(f"[ckpt] opt_step {opt_step} 断点已落盘"
                      f"({dt_ck:.0f}s)", flush=True)

    if _carto_f is not None:                 # 尾批落盘
        for m_, ls_ in _carto_buf:
            for j_, v_ in enumerate(ls_):
                _carto_f.write(json.dumps({
                    "video_id": full.recs[draw(m_ * DP * BS + j_)]["video_id"],
                    "micro": m_, "loss": round(v_, 4)}) + "\n")
        _carto_f.close()
    try:
        ms = jax.local_devices()[0].memory_stats()
        print(f"[hbm] dev0 peak={ms.get('peak_bytes_in_use', 0)/2**30:.2f}G "
              f"limit={ms.get('bytes_limit', 0)/2**30:.2f}G")
    except Exception:  # noqa: BLE001
        pass
    if pf:
        pf.close()
    flat = jax.tree_util.tree_flatten_with_path(train)[0]
    _pl = {_path_str(p): np.asarray(v) for p, v in flat}
    if a.rank_scheme == "map":               # 折叠标记随产物走(防误判 prod)
        _pl["__svd_scale_folded__"] = np.array(1)
    np.savez(os.path.join(a.out, "train_params.npz"), **_pl)
    from jax_impl.logtee import code_version
    json.dump({"loss_history": hist, "rank": a.rank, "dp": DP,
               "best_val": list(best), "seed": a.seed,
               "lr_schedule": a.lr_schedule, "warmup": a.warmup,
               "resumed_from_step": _ck["opt_step"] if _ck else None,
               "code_commit": code_version()},
              open(os.path.join(a.out, "train_meta.json"), "w"))
    has_best = os.path.exists(os.path.join(a.out, "train_params_best.npz"))
    print(f"[save] {a.out} (loss {hist[0]:.3f} -> {hist[-1]:.3f}, "
          f"best_val={best[0]:.4f}@{best[1]})"
          + ("\n[save] 评测/交付请用 train_params_best.npz"
             f"(val 最优 @step {best[1]});train_params.npz 是最后一步"
             if has_best else ""))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
