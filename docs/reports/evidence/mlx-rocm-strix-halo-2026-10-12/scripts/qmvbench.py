"""qmvbench.py: time mx.quantized_matmul (decode, M=1) per Qwen3-8B shape without a profiler; GB/s vs the 238 GB/s floor."""
import time, mlx.core as mx
SHAPES = {"k/v": (1024, 4096), "q/o": (4096, 4096), "gate/up": (12288, 4096), "down": (4096, 12288), "lm_head": (151936, 4096)}
for name, (N, K) in SHAPES.items():
    ws = [mx.quantize(mx.random.normal((N, K)).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(4)]  # 4 copies: defeat cache
    x = mx.random.normal((1, K)).astype(mx.bfloat16)
    f = lambda w: mx.quantized_matmul(x, w[0], w[1], w[2], transpose=True, group_size=64, bits=4)
    mx.eval(*ws, x)
    for _ in range(5): mx.eval(f(ws[0]))
    reps = 400 if N < 100000 else 40
    t = time.perf_counter(); mx.eval([f(ws[i % 4]) for i in range(reps)]); dt = (time.perf_counter() - t) / reps
    nbytes = N * K // 2 + (N * K // 64) * 4  # 4-bit weights + bf16 scale + bf16 bias per group
    print(f"{name:8s} N={N:6d} K={K:5d} {dt*1e6:8.1f} us  {nbytes/dt/1e9:6.1f} GB/s  floor {nbytes/238e9*1e6:7.1f} us")
