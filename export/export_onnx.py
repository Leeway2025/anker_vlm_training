"""Vision Encoder + Projector → ONNX(客户 rknn-toolkit2 编译源)。

输入固定 (1, 3, 384, 384) —— 每帧独立前向,16 帧循环 16 次
(RKNN 静态 shape 友好;帧循环在端侧 C 代码里做)。

用法:
  python -m export.export_onnx --base <hf 路径> \
      --vision-merged deliverables/vision_merged.pt --out vision_video.onnx
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--vision-merged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    import torch, yaml
    import torch.nn as nn
    from training.common import load_model_and_processor
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    cfg["model"]["name_or_path"] = a.base
    model, _ = load_model_and_processor(cfg)
    merged = torch.load(a.vision_merged, map_location="cpu")
    missing, unexpected = model.load_state_dict(merged, strict=False)
    print(f"loaded merged vision ({len(merged)} tensors)")

    # 定位 vision tower 与 projector 子模块
    vt = proj = None
    for name, mod in model.named_modules():
        low = name.lower()
        if vt is None and any(k in low for k in cfg["freeze"]["vision_keywords"]):
            vt = mod
        if any(k in low for k in cfg["freeze"]["projector_keywords"]):
            proj = mod
    assert vt is not None and proj is not None, "vision/projector 未找到"

    class VisionExport(nn.Module):
        def __init__(self, vt, proj):
            super().__init__()
            self.vt, self.proj = vt, proj

        def forward(self, pixel_values):          # (1,3,384,384) normalized
            feats = self.vt(pixel_values)
            if hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state
            return self.proj(feats)               # (1, 70, hidden)

    wrapper = VisionExport(vt, proj).eval().float()
    dummy = torch.randn(1, 3, cfg["sampling"]["image_size"],
                        cfg["sampling"]["image_size"])
    torch.onnx.export(wrapper, dummy, a.out, opset_version=a.opset,
                      input_names=["pixel_values"],
                      output_names=["visual_embeds"])
    print(f"exported -> {a.out}")


if __name__ == "__main__":
    main()
