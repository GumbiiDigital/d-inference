#!/usr/bin/env python3
"""Reduce rocprofv3 kernel traces (decode1 vs decode129) to per-token kernel cost and qmv breakdown."""
from __future__ import annotations

import csv
import gzip
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PROFILE = ROOT / "results" / "qwen3-8b-profile"
BW_GB_S = 238.0
LAYERS = 36
STRIDE = 253
TOKENS = 100


def load_trace(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    dur: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            ns = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            name = row["Kernel_Name"]
            dur[name] += ns
            cnt[name] += 1
    return dur, cnt


def short_kernel(name: str) -> str:
    s = name.replace("hip_bfloat16", "bf16").replace("unsigned char", "u8")
    if "qmv_wide_kernel" in s:
        i = s.index("qmv_wide_kernel")
        j = s.index(">", i) + 1
        return s[i:j].replace(", ", ",")
    if len(s) > 72:
        return s[:69] + "..."
    return s


def load_qmv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            if "qmv_wide_kernel" in row["Kernel_Name"]:
                rows.append(row)
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    return rows


def floor_us(n: int, k: int) -> float:
    nbytes = n * k / 2 + n * k / 64 * 4
    return nbytes / (BW_GB_S * 1e9) * 1e6


def main() -> None:
    out_path = PROFILE / "prof_reduce-output.txt"
    lines: list[str] = []

    d1, c1 = load_trace(PROFILE / "kernel_trace_decode1.csv.gz")
    d129, c129 = load_trace(PROFILE / "kernel_trace_decode129.csv.gz")

    names = set(d1) | set(d129)
    rows_a: list[tuple[float, float, str]] = []
    for name in names:
        ms = (d129.get(name, 0.0) - d1.get(name, 0.0)) / 128 / 1e6
        calls = (c129.get(name, 0) - c1.get(name, 0)) / 128
        if ms > 0 or calls > 0:
            rows_a.append((ms, calls, name))
    rows_a.sort(key=lambda x: -x[0])
    total_ms = sum(ms for ms, _, _ in rows_a)

    lines.append(f"GPU kernel ms/token total {total_ms:.2f}")
    for ms, calls, name in rows_a[:10]:
        pct = 100.0 * ms / total_ms if total_ms else 0.0
        lines.append(
            f" {ms:7.3f} ms {pct:5.1f}% calls/tok {calls:6.1f}  {short_kernel(name)}"
        )
    lines.append("")

    qmv = load_qmv_rows(PROFILE / "kernel_trace_decode129.csv.gz")
    need = STRIDE * TOKENS
    if len(qmv) < need:
        sys.exit(f"qmv_wide rows {len(qmv)} < {need}")
    block = qmv[-need:]
    for i, row in enumerate(block):
        if i % STRIDE == STRIDE - 1:
            gx = int(row["Grid_Size_X"])
            assert gx == 607744, f"lm_head Grid_Size_X at {i}: {gx}"

    med_us: dict[int, float] = {}
    n_at: dict[int, int] = {}
    for j in range(7):
        durs: list[float] = []
        ns_val: list[int] = []
        for t in range(TOKENS):
            for layer in range(LAYERS):
                idx = t * STRIDE + layer * 7 + j
                r = block[idx]
                durs.append(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]))
                ns_val.append(int(r["Grid_Size_X"]) // 4)
        med_us[j] = statistics.median(durs) / 1e3
        n_at[j] = statistics.median(ns_val)

    lm_durs = [
        int(block[t * STRIDE + STRIDE - 1]["End_Timestamp"])
        - int(block[t * STRIDE + STRIDE - 1]["Start_Timestamp"])
        for t in range(TOKENS)
    ]
    med_us[7] = statistics.median(lm_durs) / 1e3
    n_at[7] = int(block[STRIDE - 1]["Grid_Size_X"]) // 4

    lines.append(
        "qmv_wide_kernel medians (last 100 decode tokens, 36 layers); N = Grid_Size_X/4"
    )
    labels = [
        "pos0 q/k/v",
        "pos1 q/k/v",
        "pos2 q/o",
        "pos3 q/o",
        "pos4 gate/up",
        "pos5 gate/up",
        "pos6 down",
        "lm_head",
    ]
    for j, lab in enumerate(labels):
        lines.append(f"  {lab}: N={int(n_at[j if j < 7 else 7]):6d}  median {med_us[j if j < 7 else 7]:7.1f} us")
    lines.append("")

    lines.append(f"roofline floor @ {BW_GB_S:.0f} GB/s (bytes = N*K/2 + N*K/64*4); pct = floor/median*100")
    k_for = [4096] * 6 + [12288]
    total_floor_us = 0.0
    loss_small_us = 0.0
    loss_large_us = 0.0
    loss_lm_us = 0.0
    for j in range(7):
        n = int(n_at[j])
        k = k_for[j]
        fl = floor_us(n, k)
        med = med_us[j]
        pct = 100.0 * fl / med if med else 0.0
        lines.append(
            f"  pos{j}: N={n:6d} K={k:5d}  floor {fl:7.2f} us  median {med:7.1f} us  floor/median {pct:5.1f}%"
        )
        total_floor_us += LAYERS * fl
        excess = max(0.0, med - fl)
        if j <= 3:
            loss_small_us += LAYERS * excess
        else:
            loss_large_us += LAYERS * excess
    n_lm = int(n_at[7])
    fl_lm = floor_us(n_lm, 4096)
    med_lm = med_us[7]
    pct_lm = 100.0 * fl_lm / med_lm if med_lm else 0.0
    lines.append(
        f"  lm_head: N={n_lm:6d} K= 4096  floor {fl_lm:7.2f} us  median {med_lm:7.1f} us  floor/median {pct_lm:5.1f}%"
    )
    total_floor_us += fl_lm
    loss_lm_us = max(0.0, med_lm - fl_lm)

    lines.append("")
    lines.append(
        f"per-token total floor {total_floor_us / 1e3:.3f} ms; "
        f"loss above floor (median−floor): small(pos0-3) {loss_small_us / 1e3:.3f} ms, "
        f"large(pos4-6) {loss_large_us / 1e3:.3f} ms, lm_head {loss_lm_us / 1e3:.3f} ms"
    )

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
