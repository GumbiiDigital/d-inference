#!/usr/bin/env python3
"""CPU self-check for concat_qlinear with GQA-sized Q/K/V (no full model)."""

import mlx.core as mx
import mlx.nn as nn

from fuse_qwen3 import concat_qlinear


def _make_qlinear(n_out: int) -> nn.QuantizedLinear:
    lin = nn.Linear(256, n_out, bias=False)
    lin.set_dtype(mx.bfloat16)  # model dtype; MLX-ROCm qmv is wrong for float32 (see qmmcheck.py)
    return nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)


def main() -> None:
    mx.set_default_device(mx.cpu)

    q, k, v = _make_qlinear(128), _make_qlinear(32), _make_qlinear(32)
    x = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
    fused = concat_qlinear([q, k, v])
    y = fused(x)
    yq, yk, yv = mx.split(y, [128, 160], axis=-1)
    assert mx.array_equal(yq, q(x))
    assert mx.array_equal(yk, k(x))
    assert mx.array_equal(yv, v(x))

    wrong = concat_qlinear([k, q, v])
    yw = wrong(x)
    wq, wk, wv = mx.split(yw, [128, 160], axis=-1)
    assert not (
        mx.array_equal(wq, q(x))
        and mx.array_equal(wk, k(x))
        and mx.array_equal(wv, v(x))
    )

    bq, bk, bv = mx.split(y, [160, 192], axis=-1)
    assert not (
        mx.array_equal(bq, q(x))
        and mx.array_equal(bk, k(x))
        and mx.array_equal(bv, v(x))
    )

    lin_bias = nn.Linear(256, 32, bias=True)
    lin_bias.set_dtype(mx.bfloat16)
    biased = nn.QuantizedLinear.from_linear(lin_bias, group_size=64, bits=4)
    try:
        concat_qlinear([biased])
    except ValueError:
        pass
    else:
        raise AssertionError("concat_qlinear must reject layers with Linear bias")

    print("check_fuse: ok")


if __name__ == "__main__":
    main()
