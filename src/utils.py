"""
Common utilities: random input, percentiles, CSV writing
"""

import os, csv, statistics as st
import numpy as np
import torch

IM_SIZE = 224

def random_tensor_torch():
    return torch.randn(1, 3, IM_SIZE, IM_SIZE)

def random_tensor_np():
    return np.random.randn(1, 3, IM_SIZE, IM_SIZE).astype(np.float32)

def p50_p95(times_ms):
    p50 = st.median(times_ms)
    times_ms_sorted = sorted(times_ms)
    idx95 = max(0, int(0.95 * (len(times_ms_sorted)-1)))
    return p50, times_ms_sorted[idx95]

def append_metrics(path, row, header=("backend","format","size_mb","fps","p50_ms","p95_ms")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)