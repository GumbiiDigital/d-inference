"""fusebench.py: separate vs fused decode qmv (M=1) for Qwen3-8B q/k/v and gate/up. 8 weight copies rotate to defeat caches."""
import time, mlx.core as mx
def mk(N, K, c=8): return [mx.quantize(mx.random.normal((N, K)).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(c)]
def qmv(x, w): return mx.quantized_matmul(x, w[0], w[1], w[2], transpose=True, group_size=64, bits=4)
def bench(fn, reps=200):
    for i in range(5): mx.eval(fn(i))
    t = time.perf_counter(); mx.eval([fn(i) for i in range(reps)]); return (time.perf_counter() - t) / reps * 1e6
x = mx.random.normal((1, 4096)).astype(mx.bfloat16); mx.eval(x)
q, k, v, qkv = mk(4096, 4096), mk(1024, 4096), mk(1024, 4096), mk(6144, 4096)
g, u, gu = mk(12288, 4096), mk(12288, 4096), mk(24576, 4096)
mx.eval(q, k, v, qkv, g, u, gu)
print("q+k+v separate %.1f us, fused qkv %.1f us" % (bench(lambda i: [qmv(x, q[i % 8]), qmv(x, k[i % 8]), qmv(x, v[i % 8])]), bench(lambda i: qmv(x, qkv[i % 8]))))
print("gate+up separate %.1f us, fused %.1f us" % (bench(lambda i: [qmv(x, g[i % 8]), qmv(x, u[i % 8])]), bench(lambda i: qmv(x, gu[i % 8]))))
