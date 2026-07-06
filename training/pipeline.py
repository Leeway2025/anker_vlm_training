"""阶段助手: 客户逐阶段手动执行,工具只负责导航,不自动串行。

  python -m training.pipeline status        # 各阶段状态 + 建议的下一步
  python -m training.pipeline next          # 打印下一阶段的命令 + 验收标准
  python -m training.pipeline cmd <stage>   # 打印任意阶段的命令(不执行)
  python -m training.pipeline run <stage>   # 执行单个阶段,跑完即停

设计原则:
  - 一次只做一个阶段;跑完后客户看指标(每阶段验收标准会打印),
    自己决定 继续 / 重跑 / 跳过(pipeline.yaml 设 enabled: false)
  - init_from 基于磁盘真实状态缝合: 取执行顺序里最近一个"已完成"
    (outputs/<stage>/final 存在)的上游;都没有则用
    start_from_checkpoint;再没有则从 base 模型开始
  - 依赖硬校验: aux 需资产 A,cot 需资产 C,缺失时报错不静默
"""
import argparse
import glob
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGE_ORDER = ["sft_warmup", "sft_joint", "hard_mining",
               "aux_heads", "implicit_cot", "kto", "swa"]

OUT = {
    "sft_warmup": "outputs/phase5_sft_a",
    "sft_joint": "outputs/phase5_sft_b",
    "hard_mining": "outputs/phase5_sft_hm",
    "aux_heads": "outputs/phase5b_aux",
    "implicit_cot": "outputs/phase5d_cot",
    "kto": "outputs/phase6_kto",
    "swa": "outputs/swa_final",
}

# 每阶段验收标准(training_plan / REPRODUCE 摘要,跑完必看)
ACCEPT = {
    "sft_warmup": "loss 平稳下降;final/projector.pt 已生成",
    "sft_joint": "SubKS/RT 明显高于基线;loss_cls 曲线在降"
                 "(不能只有 loss_desc 降);final/ 含 adapter+projector",
    "hard_mining": "监控集上 h/n、k/l、C/D 混淆对减少;总指标不降",
    "aux_heads": "RT 相对上一阶段 +1 以上;辅助头 loss 正常下降",
    "implicit_cot": "分类指标不降;⚠️ think 泄漏率必须 = 0%"
                    "(eval.metrics 的 think_leak_rate)",
    "kto": "监控集分类不降(任一降 >0.5 应已自动停);热区混淆减少",
    "swa": "测试集指标 ≥ KTO 最后 checkpoint;产物可直接进导出",
}


def _load():
    pcfg = yaml.safe_load(open("configs/pipeline.yaml", encoding="utf-8"))
    base = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    return pcfg, base


def _final(stage, exists=os.path.exists):
    d = OUT[stage] if stage != "swa" else OUT["swa"]
    f = d if stage == "swa" else f"{d}/final"
    return f if exists(f) else None


def stage_status(pcfg, exists=os.path.exists):
    """返回 {stage: done|pending|skipped}(纯逻辑,可测)。"""
    st = {}
    for s in STAGE_ORDER:
        if not pcfg["stages"].get(s, {}).get("enabled", False):
            st[s] = "skipped"
        elif _final(s, exists):
            st[s] = "done"
        else:
            st[s] = "pending"
    return st


def init_for(stage, pcfg, exists=os.path.exists):
    """该阶段的 init_from = 执行顺序里最近的已完成上游(磁盘真实状态)。"""
    idx = STAGE_ORDER.index(stage)
    for s in reversed(STAGE_ORDER[:idx]):
        f = _final(s, exists)
        if f:
            return f
    return pcfg.get("start_from_checkpoint")


def build_cmds(stage, pcfg, data_cfg, exists=os.path.exists):
    """生成该阶段的命令列表(纯逻辑,可测)。"""
    ncores = pcfg.get("num_cores", 1)
    sp = "python" if ncores <= 1 else \
        f"python -m torch_xla.distributed.xla_spawn --num_cores {ncores}"
    prev = init_for(stage, pcfg, exists)
    out = OUT[stage]
    init = f" --init-from {prev}" if prev else ""

    if stage == "sft_warmup":
        return [f"{sp} training/train.py --phase configs/phase5_sft.yaml"
                f" --stage a{init} --output {out}"]
    if stage == "sft_joint":
        return [f"{sp} training/train.py --phase configs/phase5_sft.yaml"
                f" --stage b{init} --output {out}"]
    if stage == "hard_mining":
        if not prev:
            raise ValueError("hard_mining 需要上游检查点(先完成 sft_joint,"
                             "或设 start_from_checkpoint)")
        return [
            f"python training/run_inference.py --ckpt {prev}"
            f" --labels {data_cfg['labels_file']} --out outputs/preds_hm.jsonl",
            f"python -m training.hard_mining --preds outputs/preds_hm.jsonl"
            f" --labels {data_cfg['labels_file']} --out outputs/sw.json",
            f"{sp} training/train.py --phase configs/phase5_sft.yaml"
            f" --stage b --init-from {prev} --output {out}"
            f" --sample-weights outputs/sw.json",
        ]
    if stage == "aux_heads":
        if not exists(data_cfg.get("attributes_file", "")):
            raise ValueError(
                f"aux_heads 需要属性文件 {data_cfg.get('attributes_file')}"
                f"(Gemini 资产 A)— 先跑标注,或设 enabled: false 跳过")
        return [f"{sp} training/train.py --phase configs/phase5b_aux.yaml"
                f"{init} --output {out}"]
    if stage == "implicit_cot":
        if not exists(data_cfg.get("reasoning_file", "")):
            raise ValueError(
                f"implicit_cot 需要推理链文件 {data_cfg.get('reasoning_file')}"
                f"(Gemini 资产 C)— 先跑标注,或设 enabled: false 跳过")
        return [f"{sp} training/train.py --phase configs/phase5d_cot.yaml"
                f"{init} --output {out}"]
    if stage == "kto":
        if not prev:
            raise ValueError("kto 需要上游检查点")
        return [
            f"python training/run_inference.py --ckpt {prev}"
            f" --labels {data_cfg['labels_file']} --out outputs/preds_kto.jsonl",
            f"python -m training.build_kto_data --preds outputs/preds_kto.jsonl"
            f" --labels {data_cfg['labels_file']}"
            f" --whitelist {data_cfg['whitelist_file']}"
            f" --out outputs/kto_data.jsonl",
            f"{sp} training/kto.py --config configs/phase6_kto.yaml"
            f" --init-from {prev} --output {out}",
        ]
    if stage == "swa":
        if not prev:
            raise ValueError("swa 需要上游阶段的 checkpoints")
        src = os.path.dirname(prev)
        return [f"python -m training.swa --ckpts '{src}/checkpoint-*'"
                f" --out {out}"]
    raise ValueError(f"unknown stage {stage}")


def cmd_status():
    pcfg, base = _load()
    st = stage_status(pcfg)
    print("=== 阶段状态 ===")
    marks = {"done": "✅", "pending": "⬜", "skipped": "⏭️ "}
    for s in STAGE_ORDER:
        extra = ""
        if st[s] == "done":
            extra = f"  → {_final(s)}"
        print(f"  {marks[st[s]]} {s:14s} {st[s]}{extra}")
    nxt = next((s for s in STAGE_ORDER if st[s] == "pending"), None)
    if nxt:
        print(f"\n下一步建议: python -m training.pipeline next"
              f"(将给出 [{nxt}] 的命令)")
    else:
        last = next((s for s in reversed(STAGE_ORDER) if st[s] == "done"),
                    None)
        print(f"\n全部启用阶段已完成"
              + (f",最终产物: {_final(last)}" if last else ""))


def cmd_show(stage):
    pcfg, base = _load()
    cmds = build_cmds(stage, pcfg, base["data"])
    prev = init_for(stage, pcfg)
    print(f"=== [{stage}]  init_from={prev or 'base 模型'} ===")
    for c in cmds:
        print(f"  $ {c}")
    print(f"\n跑完后的验收标准: {ACCEPT[stage]}")
    print("通过 → 跑 status 看下一步;不通过 → 两个选择:")
    print(f"  重训: 删 {OUT[stage]} 后调参重跑")
    if stage not in ("hard_mining", "kto", "swa"):
        print(f"  追加训练(曲线还在涨): 上述命令加"
              f" --init-from {OUT[stage]}/final --epochs N")
    print("提示: epochs 是上限预算,早停(patience=3 次 eval 无改善)"
          "会自动提前结束,final 取验证集最优 checkpoint")


def cmd_next():
    pcfg, _ = _load()
    st = stage_status(pcfg)
    nxt = next((s for s in STAGE_ORDER if st[s] == "pending"), None)
    if nxt is None:
        print("没有待执行阶段(全部完成或被跳过)")
        return
    cmd_show(nxt)


def cmd_run(stage):
    pcfg, base = _load()
    cmds = build_cmds(stage, pcfg, base["data"])
    for c in cmds:
        print(f">>> {c}")
        r = subprocess.run(c, shell=True)
        if r.returncode != 0:
            print(f"⛔ 失败(exit {r.returncode})")
            sys.exit(r.returncode)
    print(f"\n✅ [{stage}] 完成。验收标准: {ACCEPT[stage]}")
    print("验收通过后跑: python -m training.pipeline status")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="op", required=True)
    sub.add_parser("status")
    sub.add_parser("next")
    p = sub.add_parser("cmd")
    p.add_argument("stage", choices=STAGE_ORDER)
    p = sub.add_parser("run")
    p.add_argument("stage", choices=STAGE_ORDER)
    a = ap.parse_args()
    if a.op == "status":
        cmd_status()
    elif a.op == "next":
        cmd_next()
    elif a.op == "cmd":
        cmd_show(a.stage)
    elif a.op == "run":
        cmd_run(a.stage)


if __name__ == "__main__":
    main()
