"""输入管线吞吐预检(开训前 5 分钟必跑,尤其 1M 轮)。

TPU 步时 ~1.5-3s(bs4×8 芯)→ 需求 ≈ 32 样本/步 ≈ 10-20 样本/s。
本脚本纯 CPU 实测 dataset 取样 + collate 吞吐,给出 worker 数建议;
供给 < 需求 → TPU 会空转,先扩 num_workers/换本地盘再开训。

用法: python scripts/bench_dataloader.py [--n 32] [--phase configs/phase5_sft.yaml]
"""
import argparse
import json
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--phase", default="configs/phase5_sft.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    phase = yaml.safe_load(open(a.phase, encoding="utf-8"))

    from transformers import AutoProcessor
    from data.build_dataset import AnkerVideoDataset, AnkerCollator
    recs = [json.loads(l) for l in
            open(cfg["data"]["labels_file"], encoding="utf-8")][:a.n]
    ds_cls = AnkerVideoDataset
    if recs and (recs[0].get("meta") or {}).get("storage") == "wds":
        from data.euno_wds import EunoWDSDataset as ds_cls
    ds = ds_cls(recs, cfg, phase, training=True)
    proc = AutoProcessor.from_pretrained(cfg["model"]["name_or_path"])
    coll = AnkerCollator(proc, cfg)

    t0 = time.time()
    exs = [ds[i] for i in range(len(recs))]
    t_ds = time.time() - t0
    t0 = time.time()
    bs = cfg["train"]["per_device_batch_size"]
    for i in range(0, len(exs), bs):
        coll(exs[i:i + bs])
    t_col = time.time() - t0

    n = len(recs)
    sps = n / (t_ds + t_col)
    need = 8 * bs / 1.5                     # 8 芯 × bs / 乐观步时
    workers = max(1, int(need / sps) + 1)
    print(f"dataset 取样: {n/t_ds:.1f}/s | collate: {n/t_col:.1f}/s | "
          f"单 worker 综合: {sps:.1f}/s")
    print(f"8 芯 bs{bs} 需求 ≈ {need:.0f}/s → 建议 num_workers ≥ {workers}"
          f"{'(⚠️ 超过 CPU 核数则输入管线是瓶颈,考虑预物化/更多主机)' if workers > os.cpu_count() else ''}")


if __name__ == "__main__":
    main()
