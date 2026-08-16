"""JAX 批量推理(hard_mining / KTO 数据生成 / 评测用)。

  python jax_impl/infer.py --labels labels.jsonl --layout hf_layout.json \
      --out preds.jsonl [--limit 8] [--max-new 40] [--init-npz lora.npz]

实现: 固定步数贪心解码,全程静态形状 —— 每步对 [T+max_new] padded 全长
做一次前向,读位置 T+i-1 的 logits argmax 写回。只编译一张图,逐步复用
(HF generate 在 XLA 上逐 token 重编译的教训,见 torch 侧 _XlaStepMarker)。
KV cache(默认开): prefill 一次建缓存 + 单 token 步进(与官方
sampler 同口径: attention_mask [B,1,cache_length] 按 arange<=pos 推进),
每样本 ≈ 1 次全长前向 + N 次轻量步;--no-kv-cache 回退旧全长路径
(每步全长前向 ~0.5s,A/B 对拍用)。
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EOT = 106
RT_SET, SK_SET = "ABCDE", "abcdefghijklmnopqrstu"


def constrained_pick(row, step, rt_ids, sk_ids, pipe_id,
                     rt_delta=None, sk_delta=None):
    """字母位受限解码: step0=RT 字母集内 argmax,1/3=强制 '|',
    2=SubKS 字母集内 argmax,其余自由。row: 该位置全词表 logits。
    rt_delta/sk_delta: 先验校正量 log(目标先验/训练先验)(可选)。"""
    import numpy as np
    if step == 0:
        lg = row[rt_ids] + (rt_delta if rt_delta is not None else 0.0)
        return rt_ids[int(np.argmax(lg))]
    if step in (1, 3):
        return pipe_id
    if step == 2:
        lg = row[sk_ids] + (sk_delta if sk_delta is not None else 0.0)
        return sk_ids[int(np.argmax(lg))]
    return int(np.argmax(row))


def prior_deltas(target_file, train_file, letters, key, tau=1.0,
                 exempt=""):
    """Δ[y] = τ·log(P_target[y]/P_train[y]);exempt 中的字母 Δ=0
    (如安全类 qrunj,避免自然先验极低把安全召回压没)。"""
    import json as _json
    import math
    tgt = _json.load(open(target_file, encoding="utf-8"))[key]
    trn = _json.load(open(train_file, encoding="utf-8"))[key]
    eps = 1e-6
    return [0.0 if c in exempt else
            tau * math.log((tgt.get(c, eps) + eps) / (trn.get(c, eps) + eps))
            for c in letters]


def legalize_combo(rt, sk):
    """RT×SubKS 非法组合矫正(training_plan 14.2 / 生产端 guard 同款):
    家人(A)不可能"包裹被拿走/未授权进入" → 升级为可疑人员(C)。"""
    if rt == "A" and sk in "nu":
        return "C", sk
    return rt, sk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--wds-dir", default=None,
                    help="显式指定分片目录(覆盖 labels 内 meta.wds_dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None,
                    help="'i/n' 隔行分片(0 起),配 infer_sharded.sh 多芯并行")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--init-npz", help="载入训练产物 lora+projector(缺省纯 base)")
    ap.add_argument("--rank-scheme", choices=["auto", "uniform", "prod", "map"],
                    default="auto",
                    help="auto=从 npz 自动判定(差异化 rank 即 prod;"
                         "折叠标记即 map=变秩 scale=1)")
    ap.add_argument("--no-proj", action="store_true",
                    help="不合并 npz 中的 projector(默认: npz 无 proj/ 即报错)")
    ap.add_argument("--no-constrain", action="store_true",
                    help="关闭字母位受限解码与非法组合矫正(默认开启)")
    ap.add_argument("--dump-letter-logits", action="store_true",
                    help="输出记录附 RT/SK 决策位裸子集 logits"
                         "(全类先验优化的原料;裸=未加任何先验偏移)")
    ap.add_argument("--prior-target", help="目标(自然)先验 json"
                    "(dump_priors.py 产物;与 --prior-train 成对启用校正)")
    ap.add_argument("--prior-train", help="训练集先验 json")
    ap.add_argument("--prior-tau", type=float, default=1.0,
                    help="校正温度(1=理论值,0.5~1 网格)")
    ap.add_argument("--prior-exempt", default="qrunj",
                    help="SK 豁免字母(Δ=0;默认安全关键类)")
    ap.add_argument("--prior-exempt-rt", default="",
                    help="RT 豁免字母(Δ=0;手术版示例: ABCD=只修 E)")
    ap.add_argument("--no-kv-cache", action="store_true",
                    help="回退无缓存全长前向(逐步 ~0.5s,对拍/排障用)")
    ap.add_argument("--ext-score-npz", default=None,
                    help="归因显著性 oracle:外部 token 打分 npz(attrib_saliency.py "
                         "产物,含 gi[M,n,64]+ids)。覆盖运动/范数启发式选择。")
    ap.add_argument("--ext-score-key", default="gi",
                    help="ext-score npz 里用哪个数组打分(gi=|grad·input|)")
    a = ap.parse_args()
    from jax_impl.logtee import tee_stdio
    tee_stdio(os.path.dirname(a.out) or ".", name=os.path.basename(a.out) + ".log")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from gemma import gm
    from gemma import peft as gpeft
    from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
    from gemma.gm.nn.gemma4 import _transformer as g4_tr
    from gemma.gm.nn.gemma4._transformer import PreprocessedVisionInput
    from jax_impl.data import (SftDataset, make_vision_input,
                               make_dyn_vision_input, make_ext_vision_input)
    SEG = os.environ.get("TOKEN_COMPRESS_MODE") == "dynseg"
    EXT_SC = None                              # 归因显著性 oracle 映射 id→[n,64]
    if a.ext_score_npz:
        _zsc = np.load(a.ext_score_npz)
        _ids = json.load(open(a.ext_score_npz.rsplit(".npz", 1)[0] + ".ids.json"))
        _arr = _zsc[a.ext_score_key]
        EXT_SC = {vid: _arr[i] for i, vid in enumerate(_ids)}
        print(f"[ext-score] 载入 {len(EXT_SC)} 条 {a.ext_score_key} "
              f"{_arr.shape} from {a.ext_score_npz}")

    def _mk_pvi(patches, pos_xy, counts, ex):
        # dynseg:逐样本逐帧预算 seg_counts 作为数据侧 pytree 叶流经 jit
        # (prefill/step 只编译一次、逐样本复用 → 必须走数据侧,不能读模块全局,
        #  否则会被烘死成首样本的常量 → 静默错值)。默认关时走原 PreprocessedVisionInput。
        if SEG:
            return make_dyn_vision_input(
                patches, pos_xy, counts,
                jnp.asarray(ex["seg_counts"][None], jnp.int32))
        return PreprocessedVisionInput(
            patches=patches, positions_xy=pos_xy, soft_token_counts=counts)

    # HF 语义对齐(v1 坑 7,与 train_sft/kto 同款): gm 前向会压缩视觉位
    # logits(_transformer.py remove_mm_logits),多模态输入下读 pos-1 的
    # 位置全部错位 → 首 token 常读成 EOT → output 全空。推理侧此前漏打
    # 此补丁(2026-07-21 客户现场实锤),勿删。
    g4_tr._token_utils.remove_mm_logits = \
        lambda logits, tokens, num_tokens_per_image: logits

    # ---- rank 方案判定必须先于模型构造(prod 是参数创建路径的 patch)----
    z, uni_rank, has_lora = None, 16, False
    if a.init_npz:
        from jax_impl.npz_io import (detect_rank_scheme, load_lora_strict,
                                     merge_proj_into_base)
        z = np.load(a.init_npz)
        has_lora = any(k.startswith("lora/") for k in z.files)
        if has_lora:
            det_scheme, ranks = detect_rank_scheme(z)
            scheme = a.rank_scheme
            if scheme == "auto":
                scheme = det_scheme
                rr = (sorted(set(ranks.values())) if isinstance(ranks, dict)
                      else ranks)
                print(f"[scheme] 从 npz 判定: {scheme} ranks={rr}")
            elif det_scheme == "map" and scheme != "map":
                # 折叠产物的 scale 语义不可协商 —— 按 prod/uniform 加载
                # 必然二次缩放,强制纠正
                print(f"[scheme] ⚠️ npz 是折叠变秩产物,--rank-scheme "
                      f"{scheme} 被覆盖为 map")
                scheme = "map"
            if scheme == "prod":
                from jax_impl.prod_lora import install_prod_lora
                install_prod_lora()      # 带 rsLoRA 缩放,与训练前向一致
            elif scheme == "map":
                if not isinstance(ranks, dict):
                    raise SystemExit("--rank-scheme map 但 npz 无折叠标记"
                                     "(非 --rank-map 产物),拒绝加载")
                from jax_impl.prod_lora import install_map_lora
                install_map_lora(ranks)  # 变秩查表,前向 scale=1
            else:
                uni_rank = ranks[0]
        else:
            print("[scheme] npz 无 lora/ 键 → base+projector 推理"
                  "(stage a 产物口径)")

    cfg64 = dataclasses.replace(
        gm.nn.Gemma4_E2B.config,
        vision_encoder=gemma_vision.VisionEncoder(
            use_clipped_linears=True, output_length=64))
    tok = gm.text.Gemma4Tokenizer()
    ds = SftDataset(a.labels, a.layout, tok, wds_dir=a.wds_dir,
                    max_label_len=a.max_new)
    # dynseg 用逐样本变长模板(哨兵总数=DYNSEG_TOTAL,T 恒定=seg_T),
    # tome 用单块模板(哨兵总数=TOME_TOTAL,T 恒定=tome_T),
    # 二者都不能用固定的 16 帧 ds.template(哨兵 1024≠512)。
    T = (ds.seg_T if ds.seg_mode
         else ds.tome_T if getattr(ds, "tome_mode", False)
         else len(ds.template))
    L = T + a.max_new

    if a.init_npz and has_lora:
        # prod 时 rank 入参仅占位(install_prod_lora 按路径覆盖)
        model = gm.nn.LoRA(rank=uni_rank,
                           model=gm.nn.Gemma4_E2B(text_only=False,
                                                  config=cfg64))
    else:
        model = gm.nn.Gemma4_E2B(text_only=False, config=cfg64)
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E2B_IT)
    params = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), params)
    if a.init_npz:
        # ② 训练过的 projector 必须跟着 LoRA 一起上车(旧版静默丢弃)
        params, n_proj = merge_proj_into_base(
            params, z, jnp, jnp.bfloat16, required=not a.no_proj)
        if not has_lora:
            print(f"[init] projector({n_proj}叶) from {a.init_npz}")
    if a.init_npz and has_lora:
        ex0 = ds[0]
        p0, x0, counts0 = make_vision_input([ex0["frames"]])
        _pvi0 = _mk_pvi(jnp.asarray(p0), jnp.asarray(x0), counts0, ex0)
        struct = jax.eval_shape(lambda: model.init(
            jax.random.PRNGKey(0), tokens=jnp.zeros((1, L), jnp.int32),
            images=_pvi0))
        lora_struct = gpeft.split_params(struct["params"])[1]
        # ① 严格加载: 未命中/形状不符直接报错,禁止静默填零
        lora = load_lora_strict(z, lora_struct, jnp, jnp.bfloat16)
        params = gpeft.merge_params(params, lora)
        print(f"[init] lora+proj({n_proj}叶) from {a.init_npz}")
        # ① 可学习打分头(env TOKEN_LEARN_SCORE=1):base=ckpt 不含 tok_scorer_*,
        # 训练态由 install_token_select 的 setup 补丁注册、单独存进 npz(键
        # "tok/<struct路径>",见 train_sft 存档)。此处按 struct 里 tok_scorer_*
        # 叶的真实路径,从 z 注入进 merged params(bf16,与训练前向一致)。
        # TOKEN_LEARN_TRAIN 不设 → 前向纯硬 top_k、无 STE,端侧可导出。
        if os.environ.get("TOKEN_LEARN_SCORE") == "1":
            from flax.traverse_util import flatten_dict, unflatten_dict
            fp = flatten_dict(params)
            n_tok = 0
            for path, _leaf in jax.tree_util.tree_flatten_with_path(
                    struct["params"])[0]:
                keys = tuple(getattr(kk, "key", str(kk)) for kk in path)
                if "tok_scorer" not in "/".join(keys):
                    continue
                zk = "tok/" + "/".join(keys)
                if zk not in z.files:
                    raise SystemExit(
                        f"[tok] init-npz 缺 {zk} —— 该产物未含可学习打分头?")
                fp[keys] = jnp.asarray(z[zk], jnp.bfloat16)
                n_tok += 1
            if n_tok == 0:
                raise SystemExit(
                    "[tok] TOKEN_LEARN_SCORE=1 但 struct 无 tok_scorer_* —— "
                    "install_token_select 的 setup 补丁未生效?")
            params = unflatten_dict(fp)
            print(f"[init] 可学习打分头 tok_scorer({n_tok}叶) from {a.init_npz}")
        # ②′ 学习式重采样器(env TOKEN_RESAMPLER=1):与打分头同通道 ——
        # 参数由 install_token_select 的 patched setup 注册、训练侧存进 npz
        # (键 "tok/<struct路径>",见 train_sft 存档);此处按 struct 里
        # tok_resampler_* 叶的真实路径从 z 注入 merged params(bf16,与训练
        # 前向一致)。重采样器前向本就无 STE 分支,训推同一条路。
        if os.environ.get("TOKEN_RESAMPLER") == "1":
            from flax.traverse_util import flatten_dict, unflatten_dict
            fp = flatten_dict(params)
            n_rsp = 0
            for path, _leaf in jax.tree_util.tree_flatten_with_path(
                    struct["params"])[0]:
                keys = tuple(getattr(kk, "key", str(kk)) for kk in path)
                if "tok_resampler" not in "/".join(keys):
                    continue
                zk = "tok/" + "/".join(keys)
                if zk not in z.files:
                    raise SystemExit(
                        f"[rsp] init-npz 缺 {zk} —— 该产物未含重采样器参数?"
                        "(热身产物请核对 RESAMPLER_* env 与训练时一致)")
                fp[keys] = jnp.asarray(z[zk], jnp.bfloat16)
                n_rsp += 1
            if n_rsp == 0:
                raise SystemExit(
                    "[rsp] TOKEN_RESAMPLER=1 但 struct 无 tok_resampler_* —— "
                    "install_token_select 的 setup 补丁未生效?")
            params = unflatten_dict(fp)
            print(f"[init] 学习式重采样器 tok_resampler({n_rsp}叶) "
                  f"from {a.init_npz}")

    @jax.jit
    def step_logits(par, tokens, pvi, pos):
        out = model.apply({"params": par}, tokens=tokens, images=pvi)
        lg = out.logits if hasattr(out, "logits") else out
        return lg[0, pos - 1].astype(jnp.float32)   # 全词表行(受限解码用)

    # ---- KV cache 路径(默认): prefill + 单 token 步进 ----
    import functools
    CL = L
    tfm = model.model if hasattr(model, "model") else model
    if not a.no_kv_cache:
        cache0 = tfm.config.init_cache(batch_size=1, dtype=jnp.bfloat16,
                                       cache_length=CL)
        PREF_POS = jnp.arange(T)[None]
        PREF_MASK = jnp.tril(jnp.ones((1, T, CL), jnp.bool_))

        @jax.jit
        def prefill_fn(par, toks_t, pvi, cache):
            out = model.apply({"params": par}, tokens=toks_t, images=pvi,
                              positions=PREF_POS, attention_mask=PREF_MASK,
                              cache=cache, return_last_only=True)
            return out.logits[0].astype(jnp.float32), out.cache

        @functools.partial(jax.jit, donate_argnums=(1,))
        def step_fn(par, cache, tok, pos):
            am = (jnp.arange(CL)[None, None, :] <= pos)
            out = model.apply({"params": par}, tokens=tok.reshape(1, 1),
                              positions=pos.reshape(1, 1),
                              attention_mask=am, cache=cache,
                              return_last_only=True)
            return out.logits[0].astype(jnp.float32), out.cache
        print(f"[decode] KV cache 开启: prefill {T} + 步进(cache_len={CL})")

    RT_IDS = np.asarray([tok.encode(c)[0] for c in RT_SET])
    SK_IDS = np.asarray([tok.encode(c)[0] for c in SK_SET])
    PIPE_ID = tok.encode("|")[0]
    RT_D = SK_D = None
    if a.prior_target and a.prior_train:
        RT_D = np.asarray(prior_deltas(a.prior_target, a.prior_train,
                                       RT_SET, "rt", a.prior_tau,
                                       exempt=a.prior_exempt_rt))
        SK_D = np.asarray(prior_deltas(a.prior_target, a.prior_train,
                                       SK_SET, "sk", a.prior_tau,
                                       exempt=a.prior_exempt))
        print(f"[decode] 先验校正开启 τ={a.prior_tau} 豁免={a.prior_exempt!r} "
              f"Δrt={[f'{x:+.2f}' for x in RT_D]}")
    if not a.no_constrain:
        print(f"[decode] 受限解码开启: RT∈{len(RT_IDS)} SK∈{len(SK_IDS)} "
              f"+ 非法组合矫正(A|n,A|u → C)")

    recs = ds.recs
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        recs = recs[i::n]                    # 隔行取样,与 torch 版同语义
    if a.limit:
        recs = recs[: a.limit]
    done = set()
    if os.path.exists(a.out):                # 断点续跑: 跳过已完成
        done = {json.loads(l)["video_id"] for l in open(a.out, encoding="utf-8")}
        if done:
            print(f"[resume] 跳过已完成 {len(done)} 条")
    fout = open(a.out, "a", encoding="utf-8")
    import time
    idx_of = {r["video_id"]: k for k, r in enumerate(ds.recs)}
    for n, rec in enumerate(recs):
        if rec["video_id"] in done:
            continue
        ex = ds[idx_of[rec["video_id"]]]
        patches, pos_xy, counts = make_vision_input([ex["frames"]])
        if EXT_SC is not None:
            sc = EXT_SC.get(rec["video_id"])
            if sc is None:
                raise SystemExit(f"[ext-score] 缺 {rec['video_id']} 的显著性")
            pvi = make_ext_vision_input(
                jnp.asarray(patches), jnp.asarray(pos_xy), counts,
                jnp.asarray(sc[None], jnp.float32))     # [1,n,64]
        else:
            pvi = _mk_pvi(jnp.asarray(patches), jnp.asarray(pos_xy), counts, ex)
        toks = np.zeros(L, np.int32)
        # dynseg/tome:用逐样本模板(ex["tokens"] 前 T 位),非固定 ds.template
        toks[:T] = (ex["tokens"][:T]
                    if (ds.seg_mode or getattr(ds, "tome_mode", False))
                    else ds.template)
        t0 = time.time()

        def pick(row, i):
            return (int(np.argmax(row)) if a.no_constrain
                    else constrained_pick(row, i, RT_IDS, SK_IDS, PIPE_ID,
                                          RT_D, SK_D))
        out_ids = []
        cap = {}                          # 字母位裸 logits(step0=RT, step2=SK)
        if not a.no_kv_cache:
            row, kv = prefill_fn(params, jnp.asarray(toks[None, :T]),
                                 pvi, cache0)
            row = np.asarray(row)
            for i in range(a.max_new):
                if a.dump_letter_logits and i == 0:
                    cap["rt"] = [float(x) for x in row[RT_IDS]]
                if a.dump_letter_logits and i == 2:
                    cap["sk"] = [float(x) for x in row[SK_IDS]]
                nxt = pick(row, i)
                out_ids.append(int(nxt))
                if nxt == EOT:
                    break
                if i + 1 < a.max_new:
                    row, kv = step_fn(params, kv,
                                      jnp.asarray(nxt, jnp.int32),
                                      jnp.asarray(T + i, jnp.int32))
                    row = np.asarray(row)
        else:
            for i in range(a.max_new):
                row = np.asarray(step_logits(params, jnp.asarray(toks[None]),
                                             pvi, T + i))
                if a.dump_letter_logits and i == 0:
                    cap["rt"] = [float(x) for x in row[RT_IDS]]
                if a.dump_letter_logits and i == 2:
                    cap["sk"] = [float(x) for x in row[SK_IDS]]
                nxt = pick(row, i)
                out_ids.append(int(nxt))
                toks[T + i] = nxt
                if nxt == EOT:
                    break
        txt = tok.decode([t for t in out_ids if t != EOT])
        if not a.no_constrain and txt.count("|") >= 2:
            seg = txt.split("|")
            rt2, sk2 = legalize_combo(seg[0].strip(), seg[1].strip())
            txt = "|".join([rt2, sk2] + seg[2:])
        if not txt.strip():
            n_empty = getattr(main, "_n_empty", 0) + 1
            main._n_empty = n_empty
        outrec = {"video_id": rec["video_id"], "output": txt.strip()}
        if a.dump_letter_logits:
            outrec.update(cap)
        fout.write(json.dumps(outrec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[infer] {n+1}/{len(recs)} {rec['video_id']} "
              f"({time.time()-t0:.1f}s): {txt[:60]!r}", flush=True)
    n_empty = getattr(main, "_n_empty", 0)
    if n_empty:
        print(f"⚠️ 本次有 {n_empty} 条空输出 —— 若比例高,疑似 logits 位置"
              f"错位或模型异常,勿直接拿去评测")
    print(f"[OK] -> {a.out}")


if __name__ == "__main__":
    main()
