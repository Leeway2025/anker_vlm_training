#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地 TPU judge 探针(零训练, 08-04 用户拍板"不要蒸馏"):
现成 IT 模型 + 原版 RT judge prompt, 在 Gemini 真值上考试。
考卷: gt_desc_judge_train.jsonl 分层抽样(mislabel/ok 各半),
指标: 与 Gemini 判决一致率 / 错标召回 / 误报率。
及格线(上岗当 1M 漏斗一级): 召回 >=90%(阈值可调向高召回)且误报 <=10%。
用法(容器内, TPU 空闲时):
  python scripts/local_judge_probe.py --model 26b --n 2000
"""
import argparse, json, random, re, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.gt_desc_judge import RULES, INSTR, build_prompt  # 与 Gemini 同款 prompt

MODELS = {
    "e2b": ("GEMMA4_E2B_IT", "Gemma4_E2B"),
    "e4b": ("GEMMA4_E4B_IT", "Gemma4_E4B"),
    "26b": ("GEMMA4_26B_A4B_IT", "Gemma4_26B_A4B"),
    "31b": ("GEMMA4_31B_IT", "Gemma4_31B"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="31b", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--truth", default="/data/gt_desc_judge_train.jsonl")
    ap.add_argument("--out", default="outputs/local_judge_probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    recs = [json.loads(l) for l in open(a.truth, encoding="utf-8")]
    random.Random(7).shuffle(recs)
    mis = [r for r in recs if r["verdict"] == "mislabel"][: a.n // 2]
    ok = [r for r in recs if r["verdict"] == "ok"][: a.n // 2]
    exam = mis + ok
    random.Random(1).shuffle(exam)
    print(f"[exam] {len(exam)} 条(mislabel {len(mis)} / ok {len(ok)})", flush=True)

    from gemma import gm
    ck, nn = MODELS[a.model]
    print(f"[load] {ck}(26b/31b 权重 >单芯 31G, FSDP 分 8 芯)", flush=True)
    model = getattr(gm.nn, nn)()
    shard = None
    if a.model in ("26b", "31b"):
        from kauldron import kd
        shard = kd.sharding.FSDPSharding()
    params = gm.ckpts.load_params(
        getattr(gm.ckpts.CheckpointPath, ck), sharding=shard)
    sampler = gm.text.ChatSampler(model=model, params=params,
                                  max_out_length=192)

    stats = dict(agree=0, miss=0, fp=0, parse_err=0)
    t0 = time.time()
    fout = open(os.path.join(a.out, f"probe_{a.model}.jsonl"), "w")
    for i, r in enumerate(exam):
        prompt = build_prompt(r["gt_rt"], r["desc"]) + \
            '\nAnswer with ONLY the JSON object, no other text.'
        txt = ""
        try:
            txt = sampler.chat(prompt, multi_turn=False)
            m = re.search(r'\{.*\}', txt, re.S)
            d = json.loads(m.group(0))
            v, cr = d.get("verdict"), d.get("correct_rt")
            # 与 Gemini 版同款硬闸
            if v == "mislabel":
                if cr not in ("A", "B", "C", "E") or r["gt_rt"] == "C" \
                        or cr == r["gt_rt"]:
                    v = "ok"
        except Exception as e:  # noqa: BLE001
            stats["parse_err"] += 1
            if stats["parse_err"] <= 3:      # 前3个异常打全栈(08-04: 31B全军
                import traceback              # 覆没被静默吞掉的教训)
                traceback.print_exc()
                print(f"[err#{stats['parse_err']}] {type(e).__name__}: {e}",
                      flush=True)
            v = "ok"          # 解析失败按 ok(保守, 会算进漏报)
            if stats["parse_err"] <= 20:     # 前20例存模型原始输出供排障
                with open(os.path.join(a.out, "parse_fail_raw.txt"),
                          "a", encoding="utf-8") as pf:
                    pf.write(f"--- #{i} {type(e).__name__}: {e}\n"
                             f"{(txt or '')[:800]}\n")
        gt = r["verdict"]
        if v == gt:
            stats["agree"] += 1
        elif gt == "mislabel":
            stats["miss"] += 1
        else:
            stats["fp"] += 1
        fout.write(json.dumps({"video_id": r["video_id"], "local": v,
                               "gemini": gt}) + "\n")
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(exam)} agree={stats['agree']} "
                  f"miss={stats['miss']} fp={stats['fp']} "
                  f"err={stats['parse_err']} | {el/(i+1):.2f}s/条", flush=True)
    fout.close()

    n = len(exam)
    rec = 1 - stats["miss"] / max(len(mis), 1)
    fpr = stats["fp"] / max(len(ok), 1)
    print(f"\n[verdict] model={a.model} 一致率 {100*stats['agree']/n:.1f}% | "
          f"错标召回 {100*rec:.1f}% | 误报率 {100*fpr:.1f}% | "
          f"解析失败 {stats['parse_err']}")
    print("[gate]", "✅ 上岗(漏斗一级)" if rec >= 0.90 and fpr <= 0.10
          else "❌ 不及格 → 换更大模型或调 prompt")


if __name__ == "__main__":
    main()
