#!/usr/bin/env bash
set -e
source .venv/bin/activate

python src/export_onnx.py
python src/bench_pytorch.py --threads 4 --iters 100 --warmup 20 --csv results/metrics.csv
python src/bench_onnx.py    --threads 4 --iters 100 --warmup 20 --csv results/metrics.csv
python src/quant_dynamic.py 
python src/bench_onnx.py    --threads 4 --iters 100 --warmup 20 --csv results/metrics.csv --model models/mobilenet_v2_int8_dyn.onnx

echo "[OK] benchmarks finished -> results/metrics.csv"