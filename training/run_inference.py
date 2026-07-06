"""批量推理 CLI(pipeline 的 hard_mining / kto 前置步骤)。"""
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
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
    records = [json.loads(l) for l in open(a.labels, encoding="utf-8")]
    if a.limit:
        records = records[:a.limit]
    generate_predictions(model, processor, records, cfg, a.out)


if __name__ == "__main__":
    main()
