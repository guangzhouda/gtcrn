import os
import onnx
import numpy as np
from onnx import numpy_helper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(BASE_DIR, "stream", "onnx_models", "gtcrn.onnx")

model = onnx.load(onnx_path)

items = []
total = 0
for t in model.graph.initializer:
    arr = numpy_helper.to_array(t)
    nbytes = arr.nbytes
    total += nbytes
    items.append((t.name, str(arr.dtype), arr.shape, nbytes))

items.sort(key=lambda x: x[3], reverse=True)

print("Total initializer bytes:", total, "=>", total/1024, "KiB")
print("\nTop 20 tensors:")
for name, dtype, shape, nbytes in items[:20]:
    print(f"{nbytes/1024:9.2f} KiB | {dtype:8} | {shape!s:18} | {name}")
