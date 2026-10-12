import MLX

// Default device and a GPU matmul; compare against a CPU-stream result.
print("default device:", Device.defaultDevice())
let a = MLXRandom.normal([1024, 1024], key: MLXRandom.key(0))
let b = MLXRandom.normal([1024, 1024], key: MLXRandom.key(1))
let gpu = matmul(a, b, stream: .gpu)
let cpu = matmul(a, b, stream: .cpu)
eval(gpu, cpu)
let maxDiff = abs(gpu - cpu).max().item(Float.self)
print("gpu sum:", gpu.sum().item(Float.self), "max |gpu-cpu|:", maxDiff)
print(maxDiff < 0.05 ? "SMOKE_OK" : "SMOKE_MISMATCH")
