"""批量推理 CLI(pipeline 的 hard_mining / kto 前置步骤)。"""
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from data.log_filter import install as _ilf
    _ilf()   # 屏蔽 processor_kwargs/fps 刷屏
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard", default=None,
                    help="'i/n': 只处理第 i 片(100k 级推理多进程/多机并行;"
                         "各片 --out 用不同文件名,最后 cat 合并)")
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    from training.common import load_model_and_processor, freeze_base, build_lora
    from training.train import restore_from
    from training.inference_utils import generate_predictions
    model, processor = load_model_and_processor(cfg)
    freeze_base(model, cfg)
    model = build_lora(model, cfg)
    restore_from(model, a.ckpt, inject_lora=True)
    if cfg.get("platform") == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            model.to(xm.xla_device())     # E2E 实测: 不上卡会在 CPU 上爬
        except ImportError:
            pass
    records = [json.loads(l) for l in open(a.labels, encoding="utf-8")]
    from training.inference_utils import shard_records
    records = shard_records(records, a.shard)
    if a.limit:
        records = records[:a.limit]
    generate_predictions(model, processor, records, cfg, a.out,
                         max_new_tokens=a.max_new_tokens,
                         batch_size=a.batch_size)


if __name__ == "__main__":
    main()
