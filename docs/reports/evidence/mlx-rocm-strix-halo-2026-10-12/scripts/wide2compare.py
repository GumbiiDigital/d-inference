#!/usr/bin/env python3
"""Compare wide2check NPZ outputs: OUT_DIR/wide2-old.npz vs wide2-{11,12,14,21,22,24}.npz."""
import sys
from pathlib import Path

import mlx.core as mx


def bit_identical(ref: dict, cur: dict) -> tuple[int, int]:
    keys = sorted(k for k in ref if not k.startswith("_"))
    same = sum(1 for k in keys if mx.array_equal(ref[k], cur[k]))
    return same, len(keys)


def main() -> None:
    out_dir = Path(sys.argv[1])
    ref = dict(mx.load(str(out_dir / "wide2-old.npz")))
    for cfg in ("11", "12", "14", "21", "22", "24"):
        cur = dict(mx.load(str(out_dir / f"wide2-{cfg}.npz")))
        n, total = bit_identical(ref, cur)
        print(f"{cfg} bit-identical: {n} / {total}")


if __name__ == "__main__":
    main()
