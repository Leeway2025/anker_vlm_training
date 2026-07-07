"""SWA: 最后 N 个 checkpoint 的 LoRA adapter 权重逐张量平均(training_plan 11 节)。

纯逻辑(average_tensors)用 numpy 可单测;入口用 safetensors。
用法:
  python -m training.swa --ckpts outputs/phase6_kto/checkpoint-* --out outputs/swa_final
"""
import argparse, glob, os, shutil


def average_tensors(dicts):
    """输入: list of {key: numpy/torch tensor}。键集必须一致。"""
    keys = set(dicts[0])
    for d in dicts[1:]:
        if set(d) != keys:
            raise ValueError("checkpoint key sets differ — 不能平均")
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--last-n", type=int, default=3)
    a = ap.parse_args()
    dirs = sorted(sum([glob.glob(p) for p in a.ckpts], []),
                  key=lambda d: int(d.rstrip("/").split("-")[-1]))[-a.last_n:]
    print(f"averaging {len(dirs)} checkpoints: {dirs}")
    from safetensors.torch import load_file, save_file
    fname = "adapter_model.safetensors"
    sds = [load_file(os.path.join(d, fname)) for d in dirs]
    avg = average_tensors(sds)
    # 只拷 adapter 配置,不 copytree —— checkpoint 内 optimizer.pt 每个
    # 3~4G,E2E 实测把盘写满(ENOSPC);SWA 产物只需 adapter 两件套
    os.makedirs(a.out, exist_ok=True)
    for f in ("adapter_config.json", "README.md"):
        src = os.path.join(dirs[-1], f)
        if os.path.exists(src):
            shutil.copy(src, a.out)
    save_file(avg, os.path.join(a.out, fname))
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
