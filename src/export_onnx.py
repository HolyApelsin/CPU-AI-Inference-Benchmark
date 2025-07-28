"""
Export MobileNetV2 from PyTorch to ONNX (WIP).
"""

import os, torch, torchvision as tv

def main():
    os.makedirs("models", exist_ok=True)
    model = tv.models.mobilenet_v2(
        weights = tv.models.MobileNet_V2_Weights.DEFAULT
    ).eval()
    x = torch.randn(1, 3, 224, 224)
    out_path = "models/mobilenet_v2.onnx"
    torch.onnx.export(
        model, x, out_path,
        input_names = ["input"], output_names = ["logits"],
        opset_version = 12,
        dynamic_axes = {"input": {0: "N"}, "logits": {0: "N"}}
    )
    mb = round(os.path.getsize(out_path)/1e6, 2)
    print(f"Saved: {out_path} ({mb} MB)")

if __name__ == "__main__":
    main()
