#!/usr/bin/env python3
"""Numerical parity and timing for gated_delta_cuda vs gated_delta_ops (run on CUDA Spark).

Brief test shapes: B=1, Hk=16, Hv=32, Dk=128, Dv=128.
Qwen3.5 TextModelArgs defaults in mlx-lm qwen3_5.py differ for a full model:
  linear_num_value_heads=64, linear_key_head_dim=192 (so Hv=64, Dk=192, Dv=128).
This script uses the brief shapes unless --real-shapes is added later; document only.
"""
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_ops, normalize_qk

from gdn_cuda import COUNTS, counts, gated_delta_cuda

B, Hk, Hv, Dk, Dv = 1, 16, 32, 128, 128
TS = [1, 7, 128, 2048]


def _errs(a, b):
    d = (a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    abs_e = float(mx.max(mx.abs(d)))
    denom = mx.maximum(mx.abs(b.astype(mx.float32)), 1e-8)
    rel_e = float(mx.max(mx.abs(d) / denom))
    return abs_e, rel_e


def _all_finite(*arrays):
    for arr in arrays:
        mx.eval(arr)
        if not bool(mx.all(mx.isfinite(arr.astype(mx.float32)))):
            return False
    return True


def _case(T, state_dtype, *, Hv_=Hv, Dv_=Dv, g=None, label_suffix=""):
    mx.random.seed(T + (0 if state_dtype == mx.float32 else 1))
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv_, Dv_)).astype(mx.bfloat16)
    q, k = normalize_qk(q, k, inv_scale=Dk**-0.5, eps=1e-6)
    if g is None:
        g = mx.random.uniform(0.5, 1.0, (B, T, Hv_)).astype(mx.float32)
    beta = mx.random.uniform(0.0, 1.0, (B, T, Hv_)).astype(mx.float32)
    state = mx.random.normal((B, Hv_, Dv_, Dk)).astype(state_dtype)
    y_ref, s_ref = gated_delta_ops(q, k, v, g, beta, state)
    y_cuda, s_cuda = gated_delta_cuda(q, k, v, g, beta, state)
    mx.eval(y_ref, s_ref, y_cuda, s_cuda)
    label = f"T={T} state={state_dtype}{label_suffix}"
    fail = False
    if not _all_finite(y_ref, s_ref, y_cuda, s_cuda):
        print(f"FAIL {label}: non-finite value in output or state", file=sys.stderr)
        fail = True
    if state_dtype == mx.float32 and s_cuda.dtype != state_dtype:
        print(f"FAIL {label}: state dtype {s_cuda.dtype} != input {state_dtype}", file=sys.stderr)
        fail = True
    ya, yr = _errs(y_cuda, y_ref)
    sa, sr = _errs(s_cuda, s_ref)
    print(f"{label}: y abs={ya:.4e} rel={yr:.4e}  state abs={sa:.4e} rel={sr:.4e}")
    if ya > 2e-2:
        print(f"FAIL {label}: y max abs {ya} > 2e-2", file=sys.stderr)
        fail = True
    if state_dtype == mx.float32 and sa > 1e-3:
        print(f"FAIL {label}: state max abs {sa} > 1e-3", file=sys.stderr)
        fail = True
    return fail


def _case_bf16_state(T=128):
    fb_before = sum(counts()["fallback"].values())
    mx.random.seed(42)
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    q, k = normalize_qk(q, k, inv_scale=Dk**-0.5, eps=1e-6)
    g = mx.random.uniform(0.5, 1.0, (B, T, Hv)).astype(mx.float32)
    beta = mx.random.uniform(0.0, 1.0, (B, T, Hv)).astype(mx.float32)
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.bfloat16)
    y_ref, s_ref = gated_delta_ops(q, k, v, g, beta, state)
    y_cuda, s_cuda = gated_delta_cuda(q, k, v, g, beta, state)
    mx.eval(y_ref, s_ref, y_cuda, s_cuda)
    label = "bf16 state fallback"
    fail = False
    fb_after = sum(counts()["fallback"].values())
    if fb_after <= fb_before:
        print(f"FAIL {label}: fallback count did not grow ({fb_before} -> {fb_after})", file=sys.stderr)
        fail = True
    if not _all_finite(y_ref, s_ref, y_cuda, s_cuda):
        print(f"FAIL {label}: non-finite value", file=sys.stderr)
        fail = True
    ya, yr = _errs(y_cuda, y_ref)
    print(f"{label}: y abs={ya:.4e} rel={yr:.4e}  fallback={counts()['fallback']}")
    if ya > 2e-2:
        print(f"FAIL {label}: y max abs {ya} > 2e-2", file=sys.stderr)
        fail = True
    return fail


def _case_dv40():
    Dv_ = 40
    T = 7
    return _case(T, mx.float32, Dv_=Dv_, label_suffix=f" Dv={Dv_}")


def _case_vectorized_gate():
    T = 7
    mx.random.seed(77)
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    q, k = normalize_qk(q, k, inv_scale=Dk**-0.5, eps=1e-6)
    g = mx.random.uniform(0.5, 1.0, (B, T, Hv, Dk)).astype(mx.float32)
    beta = mx.random.uniform(0.0, 1.0, (B, T, Hv)).astype(mx.float32)
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.float32)
    y_ref, s_ref = gated_delta_ops(q, k, v, g, beta, state)
    y_cuda, s_cuda = gated_delta_cuda(q, k, v, g, beta, state)
    mx.eval(y_ref, s_ref, y_cuda, s_cuda)
    label = "vectorized gate T=7"
    fail = False
    if not _all_finite(y_ref, s_ref, y_cuda, s_cuda):
        print(f"FAIL {label}: non-finite value", file=sys.stderr)
        fail = True
    ya, yr = _errs(y_cuda, y_ref)
    sa, sr = _errs(s_cuda, s_ref)
    print(f"{label}: y abs={ya:.4e} rel={yr:.4e}  state abs={sa:.4e} rel={sr:.4e}")
    if ya > 2e-2 or sa > 1e-3:
        print(f"FAIL {label}: error bounds exceeded", file=sys.stderr)
        fail = True
    return fail


def _case_t0():
    """T=0: empty time axis; both paths should return without crashing."""
    T = 0
    mx.random.seed(0)
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    g = mx.random.uniform(0.5, 1.0, (B, T, Hv)).astype(mx.float32)
    beta = mx.random.uniform(0.0, 1.0, (B, T, Hv)).astype(mx.float32)
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.float32)
    fail = False
    try:
        y_ref, s_ref = gated_delta_ops(q, k, v, g, beta, state)
        y_cuda, s_cuda = gated_delta_cuda(q, k, v, g, beta, state)
        mx.eval(y_ref, s_ref, y_cuda, s_cuda)
        print(
            f"T=0 behaviour: y shapes ops={y_ref.shape} cuda={y_cuda.shape}, "
            f"state unchanged shape={s_ref.shape}, finite={_all_finite(y_ref, s_ref, y_cuda, s_cuda)}"
        )
        if not _all_finite(y_ref, s_ref, y_cuda, s_cuda):
            print("FAIL T=0: non-finite value", file=sys.stderr)
            fail = True
    except Exception as ex:
        print(f"T=0 behaviour: raised {type(ex).__name__}: {ex}")
    return fail


def _time_ms(fn, *args, warmups=3, iters=10):
    for _ in range(warmups):
        y, s = fn(*args)
        mx.eval(y, s)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y, s = fn(*args)
        mx.eval(y, s)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def main():
    if not mx.cuda.is_available():
        print("SKIP: CUDA not available", file=sys.stderr)
        return 0
    mx.set_default_device(mx.gpu)
    COUNTS["cuda"] = 0
    COUNTS["fallback"].clear()
    print(
        "Note: Qwen3.5-9B config in qwen3_5.py uses Hv=64, Dk=192; this test uses brief Hv=32, Dk=128.",
        file=sys.stderr,
    )
    failed = False
    for T in TS:
        failed |= _case(T, mx.float32)
    failed |= _case_bf16_state()
    failed |= _case_dv40()
    failed |= _case_vectorized_gate()
    failed |= _case_t0()

    mx.random.seed(99)
    T = 1
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    g = mx.random.uniform(0.5, 1.0, (B, T, Hv)).astype(mx.float32)
    beta = mx.random.uniform(0.0, 1.0, (B, T, Hv)).astype(mx.float32)
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.float32)
    args = (q, k, v, g, beta, state)
    t_ops_1 = _time_ms(gated_delta_ops, *args)
    t_cu_1 = _time_ms(gated_delta_cuda, *args)
    T = 2048
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    g = mx.random.uniform(0.5, 1.0, (B, T, Hv)).astype(mx.float32)
    beta = mx.random.uniform(0.0, 1.0, (B, T, Hv)).astype(mx.float32)
    args = (q, k, v, g, beta, state)
    t_ops_2048 = _time_ms(gated_delta_ops, *args)
    t_cu_2048 = _time_ms(gated_delta_cuda, *args)
    print(f"timing median ms T=1:     ops={t_ops_1:.3f} cuda={t_cu_1:.3f}")
    print(f"timing median ms T=2048:  ops={t_ops_2048:.3f} cuda={t_cu_2048:.3f}")
    print(f"counts: {counts()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
