#!/usr/bin/env python3
"""MLX matmul micro-benchmark: bf16 vs 4-bit affine g64 (Qwen3.5-9B shapes)."""
import argparse, json, statistics, sys, time
from pathlib import Path

SHAPES = {
    "mlp_gate_up": (12288, 4096),
    "mlp_down": (4096, 12288),
    "attn_q": (8192, 4096),
    "attn_o": (4096, 4096),
    "lin_qkv": (8192, 4096),
    "lm_head": (248320, 4096),
    "attn_k": (1024, 4096),
    "attn_v": (1024, 4096),
}
DEFAULT_MS = [1, 2, 4, 512, 2048]
BENCH_SHAPES = {k: v for k, v in SHAPES.items() if k not in ("attn_k", "attn_v")}


def pct(sorted_us, p):
    if len(sorted_us) == 1:
        return sorted_us[0]
    k = (len(sorted_us) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(sorted_us) - 1)
    return sorted_us[f] + (k - f) * (sorted_us[c] - sorted_us[f])


def time_iters(mx, fn):
    for _ in range(5):
        y = fn()
        mx.eval(y)
    samples = []
    for _ in range(30):
        t0 = time.perf_counter_ns()
        y = fn()
        mx.eval(y)
        samples.append((time.perf_counter_ns() - t0) / 1e3)
    samples.sort()
    return {"us_med": statistics.median(samples), "us_p10": pct(samples, 10), "us_p90": pct(samples, 90)}


def w_bytes_bf16(n, k):
    return n * k * 2


def w_bytes_q4(n, k, gs=64):
    return n * k // 2 + (n * k // gs) * 2 * 2


def tflops(m, n, k, us):
    return (2 * m * n * k) / (us * 1e-6) / 1e12 if us else None


def bw_gbs(n, k, us, q4):
    b = w_bytes_q4(n, k) if q4 else w_bytes_bf16(n, k)
    return b / (us * 1e-6) / 1e9 if us else None


def bench_shape(mx, name, n, k, m):
    x = mx.random.normal((m, k), dtype=mx.bfloat16) * 0.02
    w = mx.random.normal((n, k), dtype=mx.bfloat16) * 0.02
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    mx.eval(x, w, wq, s, b)
    bf = time_iters(mx, lambda: x @ w.T)
    q4 = time_iters(mx, lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4))
    ratio = q4["us_med"] / bf["us_med"] if bf["us_med"] else None
    row = {"shape": name, "N": n, "K": k, "M": m, "bf16_us": bf["us_med"], "q4_us": q4["us_med"],
           "q4_over_bf16": ratio, "bf16_us_p10": bf["us_p10"], "bf16_us_p90": bf["us_p90"],
           "q4_us_p10": q4["us_p10"], "q4_us_p90": q4["us_p90"]}
    if m <= 4:
        row["bf16_weight_gbs"] = bw_gbs(n, k, bf["us_med"], False)
        row["q4_weight_gbs"] = bw_gbs(n, k, q4["us_med"], True)
    if m >= 512:
        row["bf16_tflops"] = tflops(m, n, k, bf["us_med"])
        row["q4_tflops"] = tflops(m, n, k, q4["us_med"])
    return row


def decode_sum(rows_m1):
    by = {r["shape"]: r for r in rows_m1}

    def layer(parts):
        s_bf, s_q4 = 0.0, 0.0
        for key, mult in parts:
            r = by[key]
            s_bf += mult * r["bf16_us"]
            s_q4 += mult * r["q4_us"]
        return s_bf, s_q4

    lin = [("lin_qkv", 1), ("attn_o", 1), ("mlp_gate_up", 2), ("mlp_down", 1)]
    full = [("attn_q", 1), ("attn_k", 1), ("attn_v", 1), ("attn_o", 1), ("mlp_gate_up", 2), ("mlp_down", 1)]
    bf = q4 = 0.0
    for _ in range(24):
        b, q = layer(lin)
        bf += b
        q4 += q
    for _ in range(8):
        b, q = layer(full)
        bf += b
        q4 += q
    h = by["lm_head"]
    bf += h["bf16_us"]
    q4 += h["q4_us"]
    return bf, q4


DECODE_KEYS = ("lin_qkv", "attn_o", "mlp_gate_up", "mlp_down", "attn_q", "attn_k", "attn_v", "lm_head")


def main():
    import mlx
    import mlx.core as mx

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shapes", default="", help="comma-separated shape names")
    ap.add_argument("--ms", default="", help="comma-separated M values")
    args = ap.parse_args()
    names = [s.strip() for s in args.shapes.split(",") if s.strip()] or list(BENCH_SHAPES)
    ms = [int(x) for x in args.ms.split(",") if x.strip()] or DEFAULT_MS
    for n in names:
        if n not in SHAPES:
            sys.exit(f"unknown shape {n!r}")

    info = None
    if hasattr(mx, "device_info"):
        try:
            info = mx.device_info()
        except Exception as e:
            info = {"error": repr(e)}

    rows, m1_cache = [], {}
    for name in names:
        n, k = SHAPES[name]
        for m in ms:
            row = bench_shape(mx, name, n, k, m)
            rows.append(row)
            if m == 1:
                m1_cache[name] = row

    for key in DECODE_KEYS:
        if key not in m1_cache:
            n, k = SHAPES[key]
            m1_cache[key] = bench_shape(mx, key, n, k, 1)

    bf_dec, q4_dec = decode_sum([m1_cache[k] for k in DECODE_KEYS])
    decode_row = {"shape": "approx matmul-only decode step", "M": 1, "bf16_us": bf_dec, "q4_us": q4_dec,
                  "q4_over_bf16": q4_dec / bf_dec if bf_dec else None}
    rows.append(decode_row)

    hdr = f"{'shape':<32} {'M':>5} {'bf16_us':>10} {'q4_us':>10} {'q4/bf16':>8} {'extra':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        extra = ""
        if "bf16_weight_gbs" in r:
            extra = f"bw {r['q4_weight_gbs']:.1f}G/s"
        elif "bf16_tflops" in r:
            extra = f"{r['q4_tflops']:.2f} TF/s"
        print(f"{r['shape']:<32} {r.get('M', ''):>5} {r['bf16_us']:>10.1f} {r['q4_us']:>10.1f} "
              f"{r['q4_over_bf16']:>8.3f} {extra:>14}")

    payload = {"mlx_version": mx.__version__, "device_info": info,
               "mlx_package_path": str(Path(mlx.__file__).resolve()),
               "shapes": names, "M_values": ms, "rows": rows}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
