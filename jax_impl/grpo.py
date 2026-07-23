"""GRPO 改良版(Dr.GRPO 无偏优势 + DAPO 动态采样/Clip-Higher,无 think)。

  python jax_impl/grpo.py --labels /data/labels_dedup.jsonl \
      --layout /data/hf_layout.json --init-npz outputs/jax_5b_v4/train_params_best.npz \
      --rounds 12 --chunk 2048 --group 6 --out outputs/jax_5b_grpo

设计(2026-07-23 定,适配"两字母定格式输出"的分类任务):
  - 策略空间 = 受限解码同款: RT 在 5 字母子集、SK 在 21 字母子集内
    重归一化采样 —— 训练与评测策略一致,格式零漂移;
  - 奖励塑形: SK 全对 +1.0、KS 父类对 +0.3、RT 对 +0.3(--reward-*),
    可选 --class-weight-json 按 GT 类加权(直接在奖励里矫正 m/E 先验);
  - Dr.GRPO: 优势 = r − 组均值(不除 std,消难度加权失真);
  - DAPO: ①动态采样 —— 组内奖励全同(全对/全错)的视频零梯度,直接
    丢弃(73% 底座下约六成样本,过滤后有效梯度密度 ~3×);
    ②Clip-Higher —— PPO 裁剪上界独立放宽(--eps-high 0.28);
  - 长度类改良(token 归一/超长惩罚)不适用: 决策位恒 2 token,不采纳;
  - KL 锚: 对 ref(= --init-npz 的 SFT 策略,冻结)在两个决策位的
    字母子集分布做精确 KL(5/21 维,便宜且低方差),防漂移;
  - 轮式结构(PPO 风格): 采样一批 → 1~2 epoch 裁剪更新 → 重采,
    行为 logprob 记录在采样时,ratio/clip 处理轮内策略漂移。

产物: <out>/train_params.npz(lora=策略 + proj 从 init 透传),
infer.py 直接可评。⚠️ 使用前门禁(纪律):
  ① 合成可分数据: 奖励应在数轮内逼近上限(机制正确性);
  ② 真数据首轮小规模(--rounds 2 --chunk 256)观察 kept 比例与 KL。
"""
import argparse
import dataclasses
import functools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.taxonomy import KS_GROUP  # noqa: E402

RT_SET = "ABCDE"
SK_SET = "abcdefghijklmnopqrstu"


# ---------------- 纯函数(可本地单测) ----------------

def shaped_reward(rt, sk, gt_rt, gt_sk, w_sk=1.0, w_ks=0.3, w_rt=0.3,
                  class_w=None):
    """塑形奖励;class_w: {gt_sk 字母: 权重}(奖励侧先验矫正)。"""
    r = 0.0
    if sk == gt_sk:
        r += w_sk
    if (sk in KS_GROUP and gt_sk in KS_GROUP
            and KS_GROUP[sk] == KS_GROUP[gt_sk]):
        r += w_ks
    if rt == gt_rt:
        r += w_rt
    if class_w:
        r *= float(class_w.get(gt_sk, 1.0))
    return r


def group_advantage(rewards):
    """Dr.GRPO: Â = r − mean(不除 std)。返回 (adv, keep)。
    keep=False ⇔ 组内奖励全同(DAPO 动态采样: 零梯度组,丢弃)。"""
    r = np.asarray(rewards, np.float64)
    adv = r - r.mean()
    return adv, bool(np.abs(adv).max() > 1e-9)


def sample_subset(logits_row, ids, temp, rng):
    """在字母子集上重归一化采样。返回 (子集内下标, 行为 logprob)。"""
    z = np.asarray(logits_row, np.float64)[ids] / max(temp, 1e-6)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    k = int(rng.choice(len(ids), p=p))
    return k, float(np.log(p[k]))


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--wds-dir", default=None)
    ap.add_argument("--init-npz", required=True, help="SFT best(策略起点+ref)")
    ap.add_argument("--rank-scheme", choices=["auto", "uniform", "prod"],
                    default="auto")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=2048,
                    help="每轮采样的视频数")
    ap.add_argument("--group", type=int, default=6, help="每视频采样数 G")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1, help="每轮更新遍数")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--eps-low", type=float, default=0.2)
    ap.add_argument("--eps-high", type=float, default=0.28,
                    help="DAPO Clip-Higher 上界")
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--reward-sk", type=float, default=1.0)
    ap.add_argument("--reward-ks", type=float, default=0.3)
    ap.add_argument("--reward-rt", type=float, default=0.3)
    ap.add_argument("--class-weight-json", default=None,
                    help='奖励类加权 {"m":1.5,...}(先验矫正内化)')
    ap.add_argument("--dp", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    from jax_impl.logtee import tee_stdio
    tee_stdio(a.out)

    import jax
    import jax.numpy as jnp
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
    from jax_impl.data import SftDataset, make_vision_input
    from jax_impl.npz_io import (detect_rank_scheme, load_lora_strict,
                                 merge_proj_into_base)

    devs = jax.devices()
    DP = a.dp or len(devs)
    rng = np.random.RandomState(a.seed)
    print(f"[env] devices={len(devs)} dp={DP} group={a.group} "
          f"chunk={a.chunk} rounds={a.rounds}")

    z = np.load(a.init_npz)
    scheme = a.rank_scheme
    if scheme == "auto":
        scheme, ranks = detect_rank_scheme(z)
        print(f"[scheme] 从 npz 判定: {scheme} ranks={ranks}")
    else:
        _, ranks = detect_rank_scheme(z)
    if scheme == "prod":
        from jax_impl.prod_lora import install_prod_lora
        install_prod_lora()

    # remat + 恒等旁路(与 train_sft 同款,坑勿删)
    if not getattr(g4_modules, "_REMAT_PATCHED", False):
        _orig_call = g4_modules.Block.__call__
        _POL = jax.checkpoint_policies.nothing_saveable

        def _mk(skip):
            def core(self, x, pos, cache, mask, pli, kvs):
                return _orig_call(self, x, pos, cache, mask, pli, kvs,
                                  skip_sliding_mask=skip)
            return nn.remat(core, policy=_POL, prevent_cse=False)
        _core = {False: _mk(False), True: _mk(True)}

        def patched(self, x, segment_pos, cache, attn_mask,
                    per_layer_input=None, kv_shared_cache=None,
                    skip_sliding_mask=False):
            return _core[bool(skip_sliding_mask)](
                self, x, segment_pos, cache, attn_mask,
                per_layer_input, kv_shared_cache)
        g4_modules.Block.__call__ = patched
        g4_modules._REMAT_PATCHED = True
    if not getattr(gv_tr, "_REMAT_PATCHED", False):
        gv_tr.VisionBlock = nn.remat(
            gv_tr.VisionBlock, policy=jax.checkpoint_policies.nothing_saveable)
        gv_tr._REMAT_PATCHED = True
    g4_tr._token_utils.remove_mm_logits = \
        lambda logits, tokens, num_tokens_per_image: logits

    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    tok = gm.text.Gemma4Tokenizer()
    ds = SftDataset(a.labels, a.layout, tok, wds_dir=a.wds_dir)
    T = len(ds.template)
    L = T + 4                                    # rt | sk + 1 缓冲
    model = gm.nn.LoRA(rank=ranks[0],
                       model=gm.nn.Gemma4_E2B(text_only=False, config=cfg64))

    RT_IDS = np.asarray([tok.encode(c)[0] for c in RT_SET])
    SK_IDS = np.asarray([tok.encode(c)[0] for c in SK_SET])
    PIPE = tok.encode("|")[0]
    class_w = (json.load(open(a.class_weight_json))
               if a.class_weight_json else None)

    base = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    base = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), base)
    base, n_proj = merge_proj_into_base(base, z, jnp, jnp.bfloat16,
                                        required=False)
    ex0 = ds[0]
    p00, x00, c00 = make_vision_input([ex0["frames"]])
    struct = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0), tokens=jnp.zeros((1, L), jnp.int32),
        images=PreprocessedVisionInput(
            patches=jnp.asarray(p00), positions_xy=jnp.asarray(x00),
            soft_token_counts=c00)))
    lora_struct = gpeft.split_params(struct["params"])[1]
    lora_bf = load_lora_strict(z, lora_struct, jnp, jnp.bfloat16)
    print(f"[init] policy=ref=SFT lora + proj({n_proj}叶) from {a.init_npz}")

    def _path_str(p):
        return "/".join(getattr(k, "key", str(k)) for k in p)
    EXPORTABLE = ("q_einsum", "kv_einsum", "attn_vec_einsum",
                  "gating_einsum", "/mlp/linear")

    def _is_trainable(pstr):
        llm = pstr.startswith("layer_") or "/layer_" in pstr
        vis = "vision_encoder" in pstr and "stacked_layers" in pstr
        return (llm or vis) and any(k in pstr for k in EXPORTABLE)

    # 策略参数 fp32(可训),ref 固定 bf16
    policy0 = jax.tree.map(lambda x: x.astype(jnp.float32), lora_bf)
    ref_lora = lora_bf                                  # 冻结引用

    tx = optax.adamw(a.lr, weight_decay=0.0)
    tx = optax.MultiSteps(tx, every_k_schedule=a.accum)
    mesh = Mesh(np.asarray(devs[:DP]), ("dp",))
    # 显式复制布局: base 来自 ckpt 加载器,自带 'devices' mesh 分片,
    # 直接喂 jit(shard_map) 会编译/运行布局不一致(真机门禁实锤)
    from jax.sharding import NamedSharding
    _REP = NamedSharding(mesh, P())
    base = jax.device_put(base, _REP)
    ref_lora = jax.device_put(ref_lora, _REP)
    policy0 = jax.device_put(policy0, _REP)
    opt_state = tx.init(policy0)

    # ---- 前向: 两个决策位的字母子集 logits(rollout 与训练共用底) ----
    def _rows(lora_h, base_p, tokens, patches, pos_xy):
        params = gpeft.merge_params(base_p, lora_h)
        pvi = PreprocessedVisionInput(
            patches=patches, positions_xy=pos_xy, soft_token_counts=c00)
        out = model.apply({"params": params}, tokens=tokens, images=pvi)
        lg = out.logits if hasattr(out, "logits") else out
        # 先切片后转型(全尺寸 [B,L,V] 转 fp32 会多吃 ~2G HBM,OOM 实锤)
        return (lg[:, T - 1].astype(jnp.float32),
                lg[:, T + 1].astype(jnp.float32))

    def _bf(tree):
        return jax.tree.map(lambda x: x.astype(jnp.bfloat16), tree)

    def infer_local(pol, base_p, tokens, patches, pos_xy):
        return _rows(_bf(pol), base_p, tokens, patches, pos_xy)
    infer_sh = jax.jit(shard_map(
        infer_local, mesh=mesh,
        in_specs=(P(), P(), P("dp"), P("dp"), P("dp")),
        out_specs=(P("dp"), P("dp")), check_rep=False))

    def loss_fn(pol, base_p, ref, rt_ids, sk_ids, tokens, patches,
                pos_xy, idx_rt, idx_sk, behav, adv):
        pol_h = jax.tree_util.tree_map_with_path(
            lambda p, x: x.astype(jnp.bfloat16) if _is_trainable(_path_str(p))
            else jax.lax.stop_gradient(x.astype(jnp.bfloat16)), pol)
        r1f, r2f = _rows(pol_h, base_p, tokens, patches, pos_xy)
        r1 = jnp.take(r1f, rt_ids, axis=-1)
        r2 = jnp.take(r2f, sk_ids, axis=-1)
        lp1 = jax.nn.log_softmax(r1 / a.temp)
        lp2 = jax.nn.log_softmax(r2 / a.temp)
        lp_rt = jnp.take_along_axis(lp1, idx_rt[:, None], 1)[:, 0]
        lp_sk = jnp.take_along_axis(lp2, idx_sk[:, None], 1)[:, 0]
        # PPO 裁剪(token 级,两个决策位;Clip-Higher 上界独立)
        obj = 0.0
        for lp, bh in ((lp_rt, behav[:, 0]), (lp_sk, behav[:, 1])):
            ratio = jnp.exp(lp - bh)
            un = ratio * adv
            cl = jnp.clip(ratio, 1 - a.eps_low, 1 + a.eps_high) * adv
            obj = obj + jnp.minimum(un, cl)
        # KL 锚: 子集分布对 ref 的精确 KL
        rr1f, rr2f = _rows(ref, base_p, tokens, patches, pos_xy)
        rr1 = jnp.take(rr1f, rt_ids, axis=-1)
        rr2 = jnp.take(rr2f, sk_ids, axis=-1)
        kl = 0.0
        for r_, rr in ((r1, rr1), (r2, rr2)):
            p_ = jax.nn.softmax(r_ / a.temp)
            kl = kl + (p_ * (jax.nn.log_softmax(r_ / a.temp)
                             - jax.nn.log_softmax(
                                 jax.lax.stop_gradient(rr) / a.temp))).sum(-1)
        loss = (-obj + a.kl_coef * kl).mean()
        return loss

    def grad_local(pol, base_p, ref, rt_ids, sk_ids, tokens, patches,
                   pos_xy, idx_rt, idx_sk, behav, adv):
        l, g = jax.value_and_grad(loss_fn)(pol, base_p, ref, rt_ids, sk_ids,
                                           tokens, patches, pos_xy, idx_rt,
                                           idx_sk, behav, adv)
        return (jax.lax.pmean(l, "dp"),
                jax.tree.map(lambda x: jax.lax.pmean(x, "dp"), g))
    grad_sh = shard_map(
        grad_local, mesh=mesh,
        in_specs=(P(),) * 5 + (P("dp"),) * 7,
        out_specs=(P(), P()), check_rep=False)

    RT_J = jnp.asarray(RT_IDS, jnp.int32)
    SK_J = jnp.asarray(SK_IDS, jnp.int32)

    @jax.jit
    def train_step(pol, opt_state, base_p, ref, rt_j, sk_j, *batch):
        l, g = grad_sh(pol, base_p, ref, rt_j, sk_j, *batch)
        up, opt_state = tx.update(g, opt_state, pol)
        return optax.apply_updates(pol, up), opt_state, l

    # ---- 数据装配 ----
    train_idx = list(ds.train_idx)
    by_i = {}

    def vin(idxs):
        exs = [by_i.setdefault(i, ds[i]) for i in idxs]
        pt, px, _cn = make_vision_input([e["frames"] for e in exs])
        return exs, (jnp.asarray(pt), jnp.asarray(px))

    def pad_dp(lst, fill):
        while len(lst) % DP:
            lst.append(fill)
        return lst

    policy, t0 = policy0, time.time()
    for rd in range(a.rounds):
        picks = [train_idx[i] for i in
                 rng.choice(len(train_idx), a.chunk, replace=False)]
        buf = []                    # (idx, idx_rt, idx_sk, behav2, adv)
        kept = tot = 0
        stat_r = []
        for k0 in range(0, len(picks), DP):
            grp = pad_dp(picks[k0:k0 + DP], picks[0])[:DP]
            exs, (pt, px) = vin(grp)
            toks = np.zeros((DP, L), np.int32)
            for j, e in enumerate(exs):
                toks[j, :T] = e["tokens"][:T]
            r1, _ = infer_sh(policy, base, jnp.asarray(toks), pt, px)
            r1 = np.asarray(r1)[:, RT_IDS]
            # 每视频采 G 个 RT → 逐 RT 求条件 SK 行(G 次前向,B=DP)
            rt_pick = [[sample_subset(r1[j], np.arange(len(RT_SET)),
                                      a.temp, rng)
                        for _ in range(a.group)] for j in range(DP)]
            # 注意: r1 已是子集行,sample_subset 直接在 5 维上采
            sk_rows = []
            for g_ in range(a.group):
                t2 = toks.copy()
                for j in range(DP):
                    t2[j, T] = RT_IDS[rt_pick[j][g_][0]]
                    t2[j, T + 1] = PIPE
                _, r2 = infer_sh(policy, base, jnp.asarray(t2), pt, px)
                sk_rows.append(np.asarray(r2)[:, SK_IDS])
            for j, i_ in enumerate(grp[:len(exs)]):
                lab = (ds.recs[i_].get("labels") or ds.recs[i_])
                gt_rt, gt_sk = lab["role_type"], lab["sub_keyscene"]
                branch, rew = [], []
                for g_ in range(a.group):
                    krt, lprt = rt_pick[j][g_]
                    ksk, lpsk = sample_subset(sk_rows[g_][j],
                                              np.arange(len(SK_SET)),
                                              a.temp, rng)
                    branch.append((krt, ksk, lprt, lpsk))
                    rew.append(shaped_reward(RT_SET[krt], SK_SET[ksk],
                                             gt_rt, gt_sk, a.reward_sk,
                                             a.reward_ks, a.reward_rt,
                                             class_w))
                adv, keep = group_advantage(rew)
                stat_r.append(float(np.mean(rew)))
                tot += 1
                if not keep:
                    continue        # DAPO 动态采样: 零梯度组丢弃
                kept += 1
                for (krt, ksk, lprt, lpsk), ad in zip(branch, adv):
                    buf.append((i_, krt, ksk, lprt, lpsk, float(ad)))
        print(f"[rollout] round {rd}: 视频 {tot} kept {kept} "
              f"({kept/max(tot,1):.0%}) branches={len(buf)} "
              f"mean_reward={np.mean(stat_r):.3f}", flush=True)
        if not buf:
            print("[rollout] 全部零梯度组(策略已确定化?)提前结束")
            break

        for ep in range(a.epochs):
            rng.shuffle(buf)
            micro = (len(buf) // DP) * DP
            if micro == 0:
                print(f"[train] round {rd}: 分支不足 DP,跳过")
                break
            l = 0.0
            for k0 in range(0, micro, DP):
                sl = buf[k0:k0 + DP]
                _, (pt, px) = vin([s[0] for s in sl])
                toks = np.zeros((DP, L), np.int32)
                for j, (i_, krt, ksk, *_r) in enumerate(sl):
                    toks[j, :T] = by_i[i_]["tokens"][:T]
                    toks[j, T] = RT_IDS[krt]
                    toks[j, T + 1] = PIPE
                    toks[j, T + 2] = SK_IDS[ksk]
                policy, opt_state, l = train_step(
                    policy, opt_state, base, ref_lora, RT_J, SK_J,
                    jnp.asarray(toks), pt, px,
                    jnp.asarray([s[1] for s in sl], jnp.int32),
                    jnp.asarray([s[2] for s in sl], jnp.int32),
                    jnp.asarray([[s[3], s[4]] for s in sl], jnp.float32),
                    jnp.asarray([s[5] for s in sl], jnp.float32))
            print(f"[train] round {rd} epoch {ep} loss={float(l):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        flat = jax.tree_util.tree_flatten_with_path(policy)[0]
        save = {"lora/" + _path_str(p): np.asarray(v) for p, v in flat}
        for f in z.files:                          # proj 透传,infer 直评
            if f.startswith("proj/"):
                save[f] = z[f]
        np.savez(os.path.join(a.out, "train_params.npz"), **save)
    print(f"[save] {a.out}/train_params.npz(最后一轮策略;评测选轮请按 "
          f"round 存档扩展)")


if __name__ == "__main__":
    main()
