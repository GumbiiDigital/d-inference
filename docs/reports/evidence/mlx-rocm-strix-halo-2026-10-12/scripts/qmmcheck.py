"""qmmcheck.py: GPU vs CPU quantized_matmul (4-bit, g64, transpose) over dtype/M/K/N; prints only wrong cases."""
import mlx.core as mx
bad = 0
for dt in (mx.float32, mx.bfloat16, mx.float16):
    for K in (256, 512, 1024, 4096):
        for N in (128, 4096):
            w = mx.random.normal((N, K)).astype(dt); wq, s, b = mx.quantize(w, group_size=64, bits=4)
            for M in (1, 2, 3, 4, 8, 16, 32, 64):
                x = mx.random.normal((M, K)).astype(dt)
                g = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
                c = mx.matmul(x.astype(mx.float32), mx.dequantize(wq, s, b, group_size=64, bits=4, stream=mx.cpu).astype(mx.float32).T, stream=mx.cpu)  # exact f32 reference
                cq = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4, stream=mx.cpu)
                cerr = (mx.abs(cq.astype(mx.float32) - c).max() / (mx.abs(c).max() + 1e-6)).item()
                err = (mx.abs(g.astype(mx.float32) - c.astype(mx.float32)).max() / (mx.abs(c.astype(mx.float32)).max() + 1e-6)).item()
                if err > 0.05:
                    bad += 1; print(f"WRONG {dt} M={M} K={K} N={N} rel_err={err:.3f} cpu_rel_err={cerr:.3f} gpu_allzero={(mx.abs(g).max().item()==0)}")
print("wrong cases:", bad)
