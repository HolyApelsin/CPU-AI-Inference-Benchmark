"""
Benchmark MobileNetV2 inference in PyTorch (WIP).
"""

import argparse, time
import torch, torchvision as tv
from utils import random_tensor_torch, p50_p95, append_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type = int, default = 4)
    ap.add_argument("--iters",   type = int, default = 100)
    ap.add_argument("--warmup",  type = int, default = 20)
    ap.add_argument("--csv",     type = str, default = "results/metrics.csv")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    model = tv.models.mobilenet_v2(
        weights = tv.models.MobileNet_V2_Weights.DEFAULT
    ).eval()

    x = random_tensor_torch()

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(x)
        
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter()-t0)*1000.0)
    
    mean_ms = sum(times)/len(times)
    fps = 1000.0/mean_ms
    p50, p95 = p50_p95(times)
    
    print(f"PyTorch FP32 | thr = {args.threads} | FPS = {fps:.2f} | p50 = {p50:.2f}ms | p95 = {p95:.2f}ms")
    append_metrics(args.csv, ["PyTorch", "FP32", "-", f"{fps:.2f}", f"{p50:.2f}", f"{p95:.2f}"])

if __name__ == "__main__":
    main()
