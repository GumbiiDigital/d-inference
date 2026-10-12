# Kernel body ported from mlx-lm's Metal gated-delta kernel (via ddalcu/mlx-serve
# src/transformer.zig, commit 6e15b046ac6b90aedd6ba834e639b4921bd3a933). The
# install() wrapper copies mlx-lm's gated_delta_update preparation logic.
#
# MIT License
#
# Copyright © 2023 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# MIT License
#
# Copyright (c) 2026 David Dalcu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.gated_delta import compute_g, compute_lower_bound_g, gated_delta_ops

GDN_CUDA_HEADER = """
__device__ __forceinline__ float gdn_warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
"""

_KERNELS: dict[bool, object] = {}
_LOGGED: set[str] = set()
_CUDA_LOGGED = False
_ORIGINAL_UPDATE = None
_PATCHED_FN = None
_INSTALL_REPORT: dict | None = None
COUNTS: dict = {"cuda": 0, "fallback": {}}


def counts() -> dict:
    return {"cuda": COUNTS["cuda"], "fallback": dict(COUNTS["fallback"])}


def patched_update():
    return _PATCHED_FN


def qwen3_5_patch_loaded() -> bool | None:
    qmod = sys.modules.get("mlx_lm.models.qwen3_5")
    if qmod is None:
        return None
    return getattr(qmod, "gated_delta_update", None) is _PATCHED_FN

def _log_path_once(reason: str) -> None:
    if reason in _LOGGED:
        return
    _LOGGED.add(reason)
    print(f"gdn_cuda: using mlx-lm gated_delta_ops ({reason})", file=sys.stderr)

def _cuda_source(vectorized: bool) -> str:
    if vectorized:
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "  g_ += Hv * Dk;"
    else:
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "  g_ += Hv;"
    return (
        "const int n = blockIdx.z * blockDim.z + threadIdx.z;\n"
        "const int b_idx = n / Hv;\n"
        "const int hv_idx = n % Hv;\n"
        "const int hk_idx = hv_idx / (Hv / Hk);\n"
        "constexpr int n_per_t = Dk / 32;\n"
        "const int T = T_arr[0];\n"
        "const int dk_idx = threadIdx.x;\n"
        "const int dv_idx = blockIdx.y * blockDim.y + threadIdx.y;\n"
        "if (dv_idx >= Dv) return;\n"
        "auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;\n"
        "auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;\n"
        "auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;\n"
        "auto y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;\n"
        "auto i_state = state_in + (n * Dv + dv_idx) * Dk;\n"
        "auto o_state = state_out + (n * Dv + dv_idx) * Dk;\n"
        "float state[n_per_t];\n"
        "for (int i = 0; i < n_per_t; ++i) state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);\n"
        f"{g_setup}\n"
        "auto beta_ = beta + b_idx * T * Hv;\n"
        "for (int t = 0; t < T; ++t) {\n"
        "  float kv_mem = 0.0f;\n"
        "  for (int i = 0; i < n_per_t; ++i) {\n"
        "    const int s_idx = n_per_t * dk_idx + i;\n"
        f"    state[i] = state[i] * static_cast<float>({g_access});\n"
        "    kv_mem += state[i] * static_cast<float>(k_[s_idx]);\n"
        "  }\n"
        "  kv_mem = gdn_warp_sum(kv_mem);\n"
        "  const float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);\n"
        "  float out = 0.0f;\n"
        "  for (int i = 0; i < n_per_t; ++i) {\n"
        "    const int s_idx = n_per_t * dk_idx + i;\n"
        "    state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;\n"
        "    out += state[i] * static_cast<float>(q_[s_idx]);\n"
        "  }\n"
        "  out = gdn_warp_sum(out);\n"
        "  if (dk_idx == 0) y_[dv_idx] = static_cast<OutT>(out);\n"
        "  q_ += Hk * Dk;\n"
        "  k_ += Hk * Dk;\n"
        "  v_ += Hv * Dv;\n"
        "  y_ += Hv * Dv;\n"
        f"{g_advance}\n"
        "  beta_ += Hv;\n"
        "}\n"
        "for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);\n"
    )

def make_kernel(vectorized: bool):
    if vectorized not in _KERNELS:
        suffix = "_vec" if vectorized else ""
        _KERNELS[vectorized] = mx.fast.cuda_kernel(
            name=f"gated_delta_step_cuda{suffix}",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T_arr"],
            output_names=["y", "state_out"],
            source=_cuda_source(vectorized),
            header=GDN_CUDA_HEADER,
            ensure_row_contiguous=True,
            shared_memory=0,
        )
    return _KERNELS[vectorized]

def gated_delta_cuda(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]

    if not mx.cuda.is_available():
        return _fallback("cuda not available", q, k, v, g, beta, state, mask)
    if mx.default_device() != mx.gpu:
        return _fallback(f"default device is {mx.default_device()}", q, k, v, g, beta, state, mask)
    if mask is not None:
        return _fallback("mask is not None", q, k, v, g, beta, state, mask)
    if Dk % 32 != 0:
        return _fallback(f"Dk={Dk} not divisible by 32", q, k, v, g, beta, state, mask)
    if Hv % Hk != 0:
        return _fallback(f"Hv % Hk != 0 ({Hv}, {Hk})", q, k, v, g, beta, state, mask)
    if state.dtype != mx.float32:
        return _fallback(
            f"state dtype {state.dtype} (kernel keeps fp32 state only)",
            q, k, v, g, beta, state, mask,
        )

    global _CUDA_LOGGED
    if not _CUDA_LOGGED:
        _CUDA_LOGGED = True
        print("gdn_cuda: using CUDA kernel", file=sys.stderr)
    kernel = make_kernel(g.ndim == 4)
    t_arr = mx.array([T], dtype=mx.int32)
    COUNTS["cuda"] += 1
    return kernel(
        inputs=[q, k, v, g, beta, state, t_arr],
        template=[
            ("StT", state.dtype),
            ("OutT", q.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )

def _fallback(reason, q, k, v, g, beta, state, mask):
    fb = COUNTS["fallback"]
    fb[reason] = fb.get(reason, 0) + 1
    _log_path_once(reason)
    return gated_delta_ops(q, k, v, g, beta, state, mask)

def install() -> dict:
    global _ORIGINAL_UPDATE, _PATCHED_FN, _INSTALL_REPORT
    import mlx_lm.models.gated_delta as gd

    if _PATCHED_FN is not None and gd.gated_delta_update is _PATCHED_FN:
        return dict(_INSTALL_REPORT or {"mlx_lm.models.gated_delta": True})

    if _ORIGINAL_UPDATE is None:
        _ORIGINAL_UPDATE = gd.gated_delta_update

    def gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True,
        lower_bound=None, allow_neg_eigval=False,
    ):
        beta = mx.sigmoid(b)
        if allow_neg_eigval:
            beta = beta * 2.0
        if lower_bound is None:
            g = compute_g(A_log, a, dt_bias)
        else:
            g = compute_lower_bound_g(A_log, a, dt_bias, lower_bound)
        if state is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        if not use_kernel:
            return gated_delta_ops(q, k, v, g, beta, state, mask)
        if mx.cuda.is_available() and mx.default_device() == mx.gpu:
            return gated_delta_cuda(q, k, v, g, beta, state, mask)
        kw = dict(use_kernel=use_kernel, lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        return _ORIGINAL_UPDATE(q, k, v, a, b, A_log, dt_bias, state, mask, **kw)

    _PATCHED_FN = gated_delta_update
    gd.gated_delta_update = gated_delta_update
    report = {"mlx_lm.models.gated_delta": True}
    for name, mod in list(sys.modules.items()):
        if not name.startswith("mlx_lm.models."):
            continue
        if hasattr(mod, "gated_delta_update"):
            mod.gated_delta_update = gated_delta_update
            report[name] = True
    _INSTALL_REPORT = report
    return report
