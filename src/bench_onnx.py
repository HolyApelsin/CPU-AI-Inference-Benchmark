"""
Benchmark MobileNetV2 inference in ONNX Runtime (FP32, CPU).
"""

import argparse, os, time
import onnxruntime as ort
from utils import random_tensor_np, p50_p95, append_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   type = str, default = "models/mobilenet_v2.onnx")
    ap.add_argument("--threads", type = int, default = 4)
    ap.add_argument("--iters",   type = int, default = 100)
    ap.add_argument("--warmup",  type = int, default = 20)
    ap.add_argument("--csv",     type = str, default = "results/metrics.csv")
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    x = random_tensor_np()

    for _ in range(args.warmup):
        _ = sess.run(None, {input_name: x})
                
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: x})
        times.append((time.perf_counter()-t0)*1000.0)
    
    mean_ms = sum(times)/len(times)
    fps = 1000.0/mean_ms
    p50, p95 = p50_p95(times)

    size_mb = round(os.path.getsize(args.model)/1e6, 2)
    print(f"ONNXRuntime FP32 | thr = {args.threads} | FPS = {fps:.2f} | p50 = {p50:.2f}ms | p95 = {p95:.2f}ms | size = {size_mb}MB")
    append_metrics(args.csv, ["ONNXRuntime", "FP32", f"{size_mb}", f"{fps:.2f}", f"{p50:.2f}", f"{p95:.2f}"])

if __name__ == "__main__":
    main()