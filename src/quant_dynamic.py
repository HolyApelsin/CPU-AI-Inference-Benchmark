"""
Dynamic INT8 quantization of ONNX model.
"""

import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    os.makedirs("models", exist_ok=True)
    src = "models/mobilenet_v2.onnx"
    dst = "models/mobilenet_v2_int8_dyn.onnx"
    quantize_dynamic(
        src, dst,
        weight_type = QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"]
        )
    mb = round(os.path.getsize(dst)/1e6, 2)
    print(f"Saved INT8 (dynamic): {dst} ({mb} MB)")

if __name__ == "__main__":
    main()