"""wide2check.py OUT.npz: fixed-seed qmv outputs (bf16/f16, 4-bit, M=1..8) for bitwise comparison of MLX_QMV_WIDE2 on/off."""
import sys, mlx.core as mx
mx.random.seed(0); res = {}
for dt in (mx.bfloat16, mx.float16):
    for gs in (32, 64, 128):
        for N, K in ((1024, 4096), (4096, 4096), (12288, 4096), (4096, 12288), (151936, 4096), (33, 256), (7, 1024)):
            wq, s, b = mx.quantize(mx.random.normal((N, K)).astype(dt), group_size=gs, bits=4)
            for M in (1, 3, 8):
                x = mx.random.normal((M, K)).astype(dt)
                res[f"{dt}_{gs}_{N}_{K}_{M}"] = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=gs, bits=4)
mx.savez(sys.argv[1], **res); print(len(res), "outputs")
