# MLX-CUDA on DGX Spark: numerics, llama.cpp ratios and Linux gaps

> Last updated: 2026-10-12

Official pip **MLX 0.32.3** with **mlx-cuda-13 0.32.3** runs Qwen checkpoints on a **DGX Spark** (GB10, sm_121) with teacher-forced logprobs as close to **M5 Max** Metal as Metal's own prefill and decode paths are to each other (**2** of **4,146** positions above **0.3** nats). Against llama.cpp on the same box, dense decode is near parity (**0.93–0.996**) and stock mlx-lm Qwen3.5 prefill is far behind (**0.10–0.19**); a local **CUDA gated-delta kernel** lifts bf16 Qwen3.5 prefill to **0.84–0.98**, while 4-bit prefill stays at **0.34–0.42** (Qwen3.5) and **0.39–0.50** (dense) after a quantized-matmul swap. No Darkbloom provider served on Linux: `libs/mlx-swift` needs local workarounds, `libs/mlx-swift-lm` needs platform guards, and 8 of 12 `provider-swift` targets stop at Apple-only dependencies.

## Results

Ratio = MLX-CUDA median ÷ llama.cpp median of **5** timed reps, **batch 1**, one box (DGX Spark A). Stock mlx-lm and the gated-delta kernel are shown in separate tables. 4-bit formats differ by engine: MLX affine 4-bit, group size **64** (bf16 scale and bias) vs GGUF **`Q4_0`**, block **32** (fp16 scale), quantized with `--pure`; both about 4.5 bits per weight, from the same bf16 source.

**Runs 1–2 — dense Qwen3-8B and Qwen3.5 (stock mlx-lm, no gdn patch)**

| Model | Prefill 512 | Prefill 2048 | Decode @512 | Decode @2048 |
|---|---|---|---|---|
| Qwen3-8B dense bf16 (run1 / run2) | 0.69 / 0.75 | 1.06 / 1.17 | 0.995 / 0.975 | 0.996 / 0.981 |
| Qwen3-8B dense 4-bit (run1 / run2) | 0.50 / 0.49 | 0.40 / 0.40 | 0.93 / 0.99 | 0.94 / 0.98 |
| Qwen3.5-9B hybrid bf16 (run1 / run2) | 0.18 / 0.19 | 0.14 / 0.15 | 0.96 / 0.95 | 0.95 / 0.94 |
| Qwen3.5-9B hybrid 4-bit (run1 / run2) | 0.12 / 0.12 | 0.10 / 0.10 | 0.85 / 0.84 | 0.84 / 0.84 |

Run 2 absolute medians (tok/s) for Qwen3-8B bf16 prefill **2086** vs **2796** (P=512) and Qwen3.5 bf16 decode **13.7** vs **14.4** (P=512) illustrate the spread. Run 2 decode CV ≤ **1.9 %** per cell; prefill CV ≤ **3.1 %** except MLX dense bf16 P=512 (**22.8 %**). Run 1 llama.cpp P=512 prefill CV reached **20.6 %** after a long→short prompt switch; run 2 added an unrecorded round to mitigate that.

Evidence: run1 JSON at [results/](evidence/mlx-cuda-dgx-spark-2026-10-12/results/bench-qwen3-8b-bf16-spark-a.json.gz) (and sibling q4/bf16/q4 q35 files); run2 under [run2/](evidence/mlx-cuda-dgx-spark-2026-10-12/results/run2/).

**Stock mlx-lm Qwen3.5 mechanism (likely, not kernel-swapped in runs 1–2).** In mlx-lm, `mlx_lm/models/gated_delta.py` uses a fused Metal kernel only when `mx.metal.is_available()`; on CUDA the gated-delta recurrence falls back to plain ops—consistent with flat ~**300–360** tok/s Qwen3.5 prefill on GB10.

**Runs 3–4 — Qwen3.5 with `--mlx-patch gdn_cuda`**

| Qwen3.5-9B | Prefill 512 | Prefill 2048 | Decode @512 | Decode @2048 |
|---|---|---|---|---|
| bf16, ops (run1 / run2) | 0.18 / 0.19 | 0.14 / 0.15 | 0.96 / 0.95 | 0.95 / 0.94 |
| bf16, CUDA kernel (run3 / run4) | 0.96 / 0.98 | 0.84 / 0.87 | 0.98 / 0.96 | 0.98 / 0.95 |
| 4-bit, ops (run1 / run2) | 0.12 / 0.12 | 0.10 / 0.10 | 0.85 / 0.84 | 0.84 / 0.84 |
| 4-bit, CUDA kernel (run3 / run4) | 0.39 / 0.40 | 0.34 / 0.34 | 0.90 / 0.90 | 0.88 / 0.91 |

Run 4 logged **43,680** kernel calls and **0** fallbacks per benchmark (`patched_qwen3_5` true). Run 4 bf16 medians (tok/s): prefill **1764** vs **1799** (512), **2093** vs **2393** (2048); decode **14.0** vs **14.6** (512).

Teacher-forced numerics with kernel (run 3, fp32/no-mask path): mean **0.0034** (prefill) and **0.0039** (decode) nats; **1** and **2** positions > **0.3** nats—same order as Metal/CUDA without kernel.

JSON: [run3](evidence/mlx-cuda-dgx-spark-2026-10-12/results/run3/), [run4](evidence/mlx-cuda-dgx-spark-2026-10-12/results/run4/).

**Run 5 — MLX built from source (pip, `base`, `qmm`) + gdn_cuda**

**Observed gap (with gdn kernel, stock pip MLX).** 4-bit Qwen3.5 prefill ratios **~0.34–0.42** (runs 3–5) vs dense 4-bit **~0.40–0.50**; decode **~0.88–0.91** vs dense **~0.93–0.99**. bf16 decode loss vs llama.cpp with kernel remains **~1.9–4.6 %** (not a demonstrated 1–2 % floor).

| 4-bit model | MLX variant | Prefill 512 | Prefill 2048 | Decode @512 | Decode @2048 |
|---|---|---|---|---|---|
| Qwen3.5-9B + gdn_cuda | pip | 0.42 | 0.35 | 0.90 | 0.89 |
| Qwen3.5-9B + gdn_cuda | base | 0.40 | 0.34 | 0.88 | 0.87 |
| Qwen3.5-9B + gdn_cuda | qmm | 0.41 | 0.34 | 0.90 | 0.89 |
| Qwen3-8B dense | pip / base / qmm | ~0.48–0.50 | ~0.39–0.41 | ~0.96–0.98 | ~0.97–0.98 |

The **mlx-cuda-qmm-read-weights-once** patch (from mlx-serve, applied to `64ea011`) moved end-to-end medians by **−1.6 % to +1.8 %** between `base` and `qmm`; llama.cpp medians also moved between sessions. Patch targets small row-count quantized matmul (2–7 rows); this harness uses **1-row** decode and **512+** row prefill.

Wheel receipts: base sha256 prefix **`3d5b11bc…`**, qmm **`38330bab…`** in the build logs ([base](evidence/mlx-cuda-dgx-spark-2026-10-12/results/run5/build-mlx-base.log.gz), [qmm](evidence/mlx-cuda-dgx-spark-2026-10-12/results/run5/build-mlx-qmm.log.gz)).

**Numerical agreement (MLX-CUDA vs M5 Metal)**

Cross-platform checks need a **logprob tolerance**, not exact token identity. Free-run greedy: **3** of **8** prompts match fully; **4** divergences sit at exact bf16 top-1/top-2 ties; **1** non-tie divergence (prompt p1, generated token 38) is one of the **2** decode-path outliers in the table below (Metal −0.067 vs CUDA −1.385 at that position).

| Comparison (4-bit Qwen3.5-9B) | mean abs. Δ logprob (per prompt) | positions > 0.3 nats |
|---|---|---|
| M5 Metal vs GB10 CUDA, prefill path | 0.001–0.004 | **0** |
| M5 Metal vs GB10 CUDA, decode path | 0.003–0.006 | **2** |
| M5 Metal prefill vs M5 Metal decode (baseline) | 0.002–0.005 | **2** |

Raw traces and TF JSON: [trace-m5-metal](evidence/mlx-cuda-dgx-spark-2026-10-12/results/trace-m5-metal-Qwen3.5-9B-MLX-4bit.jsonl.gz), [trace-spark-a-cuda-4bit](evidence/mlx-cuda-dgx-spark-2026-10-12/results/trace-spark-a-cuda-Qwen3.5-9B-MLX-4bit.jsonl.gz), [tf-m5-metal-4bit](evidence/mlx-cuda-dgx-spark-2026-10-12/results/tf-m5-metal-4bit.json.gz), [tf-spark-a-cuda-4bit](evidence/mlx-cuda-dgx-spark-2026-10-12/results/tf-spark-a-cuda-4bit.json.gz).

**Swift token match (Qwen3-8B 4-bit, same box)**

Prompt token ids are identical on all 8 prompts. No Swift speed was measured.

| Comparison (Qwen3-8B 4-bit, 8 prompts × 64 greedy tokens, same box and model folder) | Full matches | Divergences |
|---|---|---|
| Swift mlx-swift-lm (`3fd4944` + guards) + mlx-swift `6923a80`, CUDA vs Python MLX `0.32.3.dev20261012+64ea011` (source build of v0.32.3, no patch), CUDA | **7 / 8** | **1**, at an exact bf16 tie (Python top-1/top-2 margin **0.000**, prompt p4, token 12) |

Traces: [Python](evidence/mlx-cuda-dgx-spark-2026-10-12/results/trace-spark-b-python-q3-8b-q4g64.jsonl.gz), [Swift](evidence/mlx-cuda-dgx-spark-2026-10-12/results/trace-spark-b-swift-q3-8b-q4g64.jsonl.gz).

## Method And Gates

**Speed harness** (`scripts/bench_compare.py`): one Python driver; MLX in-process via `mlx_lm.generate_step`; llama.cpp as `llama-server`. Identical prompt token ids; **P** ∈ {512, 2048} prefill; **N** = 128 decode tokens; EOS ignored on the llama.cpp side; `cache_prompt: false`. Client-side `perf_counter_ns`: prefill tok/s = **P** / (first token − start); decode tok/s = **(N−1)** / (last − first). **1** warmup per engine and length, **1** unrecorded ordering round, then **5** timed reps with alternating engine order. GPU gate before each timed rep: temperature ≤ **60 °C**, utilization ≤ **10 %**.

**Teacher forcing.** `scripts/mlx_trace.py` greedy traces (8 prompts × 64 tokens) and `scripts/tf_logprobs.py` teacher forcing on the same token sequences (509–533 tokens per prompt after extension (508–532 scored positions each), **4,146** scored next-token positions). M5 used Metal with the same `mlx` / `mlx-lm` versions; GB10 used CUDA. No chat template; greedy argmax; EOS not used as a stop in the trace path.

**Weights provenance.** bf16 MLX and GGUF pairs share the same bf16 checkpoint; 4-bit pairs differ in quantization format (not bit-identical weights). One bf16 safetensor source per pair; MLX affine 4-bit g64 vs GGUF `Q4_0 --pure` for Qwen3-8B dense comparisons.

**Gated-delta CUDA kernel.** A local monkeypatch `scripts/gdn_cuda.py` (enabled with `bench_compare.py --mlx-patch gdn_cuda`) ports the mlx-serve / mlx-lm gated-delta recurrence to `mlx_fast_cuda_kernel`. Licence: full MIT notices from mlx-lm (Apple Inc.) and mlx-serve in the file header—unchanged in evidence [gdn_cuda.py](evidence/mlx-cuda-dgx-spark-2026-10-12/scripts/gdn_cuda.py). `scripts/test_gdn_cuda.py` on GB10: kernel vs `gated_delta_ops` max abs error **1.2e−4** through **T=2048**; one layer call at T=2048 (test dims Hv=32, Hk=16, Dk=Dv=128, which are the `linear_*` values in the measured checkpoints' `config.json` ([receipt](evidence/mlx-cuda-dgx-spark-2026-10-12/results/env/qwen35-9b-config-linear-attn.txt.gz)); the test script's stderr note quotes mlx-lm's `TextModelArgs` defaults, Hv=64 and Dk=192, which these checkpoints do not use) ~**197 ms** (ops) vs ~**5.5 ms** (kernel). Receipt: [test-gdn-cuda.txt](evidence/mlx-cuda-dgx-spark-2026-10-12/results/run4/test-gdn-cuda.txt.gz). The kernel runs when CUDA is the default GPU device, no mask is passed, the state is fp32, `Dk % 32 == 0` and `Hv % Hk == 0`; other cases fall back to `gated_delta_ops` and are counted (unit test also covers `Dv=40` and a vectorized gate).

**QMM micro-benchmark** (`scripts/qmm_micro.py`): at **4** rows, qmm patch cuts lm_head **10.9 → 6.4 ms** and mlp_gate_up **0.94 → 0.50 ms**; at **2048** rows 4-bit matmul is **1.36–2.21×** slower than bf16 on the same shape. The micro-benchmark printed all timings; its JSON export then failed (`TypeError` on a missing package path), so the logs are the record. Logs: [run5/qmm-micro-*.log](evidence/mlx-cuda-dgx-spark-2026-10-12/results/run5/).

## Frozen Provenance

| Component | Pin |
|---|---|
| Python venv | 3.12.3, offline wheel install (`--no-index`) |
| `mlx` | **0.32.3** wheel sha256 prefix `e75fa55161d9…` |
| `mlx-cuda-13` | **0.32.3** wheel sha256 prefix `eeb934fcbaca…` |
| `mlx-lm` | **0.32.0** wheel sha256 prefix `9a0f3843135f…` |
| CUDA runtime | 13.x from pip wheels; driver reports CUDA **13.0** on GB10 |
| llama.cpp | **llama-server** build **1**, commit **`0f701c8`**, `-fa on`, `-ngl 99` |
| MLX source (run 5) | upstream **`64ea011`** (MLX v0.32.3 tag tree) via `build-mlx-src.sh` |
| mlx-serve (read-only) | [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) @ **`6e15b046`** |

Full wheel and pip-freeze lists: [evidence/env](evidence/mlx-cuda-dgx-spark-2026-10-12/results/env/wheels-mlxcuda.SHA256SUMS.gz) and [pip-freeze-mlxcuda](evidence/mlx-cuda-dgx-spark-2026-10-12/results/env/pip-freeze-mlxcuda.txt.gz).

| Model role | Hugging Face id / local tree | Revision or hash anchor |
|---|---|---|
| Cross-platform 4-bit trace | `mlx-community/Qwen3.5-9B-MLX-4bit` | HF revision **`938d891`**; shard hashes in [sha256-mlx4bit](evidence/mlx-cuda-dgx-spark-2026-10-12/results/env/sha256-mlx4bit.txt.gz) |
| Qwen3.5 hybrid bf16 | One bf16 safetensor source → MLX + GGUF | [sha256-bf16](evidence/mlx-cuda-dgx-spark-2026-10-12/results/env/sha256-bf16.txt.gz) (prefix `db6f444b…` …) |
| Qwen3-8B dense bf16 / 4-bit | Same bf16 source per pair; MLX affine 4-bit g64 vs GGUF `Q4_0 --pure` | [sha256-qwen3-8b](evidence/mlx-cuda-dgx-spark-2026-10-12/results/env/sha256-qwen3-8b.txt.gz) |

| Label | Role |
|---|---|
| DGX Spark A | All MLX-CUDA and llama.cpp speed runs (GB10, sm_121) |
| DGX Spark B | Unpatched MLX source build for run 5 (`base`), and all Swift builds and runs; the `qmm` build ran on Spark A |
| M5 Max | Metal reference traces and teacher-forced baselines |

Built on DGX Spark B (Ubuntu 24.04 aarch64, GB10) with the official Swift **6.3.3** toolchain for Ubuntu 24.04 aarch64 (GPG signature verified, key `EF80A866B47A981F`, signed 2026-06-29), CUDA **13.0**, `CUDA_ARCH=sm_121`. Submodules as pinned at `7796fe2`: `libs/mlx-swift` **`6923a80`** (MLX core **`cb77239`**, mlx-c **`02cf6f4`**) and `libs/mlx-swift-lm` **`3fd4944`**. Script: [build-mlx-swift.sh](evidence/mlx-cuda-dgx-spark-2026-10-12/scripts/build-mlx-swift.sh). Receipts: [swift-toolchain](evidence/mlx-cuda-dgx-spark-2026-10-12/results/swift-toolchain.txt.gz), [mlx-swift build log](evidence/mlx-cuda-dgx-spark-2026-10-12/results/build-mlx-swift.log.gz).

## Linux Build Gaps

**mlx-swift builds and runs a GPU op, after local workarounds (rows 1-6 at build time, row 7 at run time).** A one-op package ([swift-smoke](evidence/mlx-cuda-dgx-spark-2026-10-12/scripts/swift-smoke/)) reports `Device(gpu, 0)` and a 1024×1024 matmul within **0.045** of the CPU result (sums of 1,024 products). The committed package does not build as is:

| # | Gap in `libs/mlx-swift` at `6923a80` | Local workaround |
|---|---|---|
| 1 | `Source/Cmlx/mlx-generated/cuda/cuda_jit_sources.h` is older than core `cb77239`: **6** `jit_source_*` symbols missing ([inventory](evidence/mlx-cuda-dgx-spark-2026-10-12/results/source-counts/mlx-swift-cuda-jit-symbols.txt.gz)) (`qmm_sm80`, `qmm_sm90`, `qmm_naive`, `gemm_sm70`, `cute_dequant`, `slice_update`). `tools/update-mlx.sh` copies it from a CMake build but runs on macOS, where the CUDA target is not configured | Regenerated with MLX's own CMake target `cuda_jit_sources` (16 → 22 symbols) |
| 2 | `cudnn_frontend.h` not found (MLX's CMake fetches cudnn-frontend; SwiftPM does not) | cudnn-frontend **v1.16.0** headers (the tag MLX pins) |
| 3 | The `CudaBuild` plugin compiles every `.cu` it finds, including qmm files that `Package.swift` excludes; they need CUTLASS/CuTe | CUTLASS **v4.4.2** headers on nvcc's path |
| 4 | System CCCL from CUDA 13.0 gives an ambiguous `cuda::std::fmod` in `backend/cuda/device/binary_ops.cuh` | CCCL **v3.1.3** (MLX's pin) on both nvcc and clang paths |
| 5 | `cufft*` undefined at link (`backend/cuda/fft.cu`); `Package.swift` does not link `cufft` | `-lcufft` |
| 6 | CUDA 13.0 rejects Swift 6.3.3's clang 21 as nvcc host compiler | `-allow-unsupported-compiler` (unsupported pair; checked numerically) |
| 7 | At run time, NVRTC JIT kernels include CuTe/CUTLASS headers that are not on NVRTC's include path | 3-line local change in `mlx/backend/cuda/jit_module.cpp` (`include_path_args`): extra path from `MLX_NVRTC_EXTRA_INCLUDE` ([patch](evidence/mlx-cuda-dgx-spark-2026-10-12/results/mlx-swift-nvrtc-include.patch.gz)) |

OpenBLAS **v0.3.33** was built from source for `blas`/`lapack`, and cuDNN came from the `nvidia-cudnn-cu13` wheel; neither is a package defect.

**mlx-swift-lm does not build on Linux at `3fd4944`.** `MLXLMCommon` imports Apple-only modules: `CoreGraphics` (`ChatSession.swift`), `CoreImage` (`UserInput.swift`) and `os` (`ContinuousBatchingV2/EngineV2.swift`, `EngineLoopV2.swift`, `Paged/PagedSelectedGather.swift`); `MLXLLM` imports `os` in seven `Qwen4Exp*` files. A local branch adds platform guards only (`#if canImport(...)`, a `LinuxCompat.swift` shim for `os.Logger`/signposts/wall time, `swift-crypto` for SHA-256 on Linux; **30 files, +344/−78**, portability edits only; not reviewed for Apple-side logging changes): [mlx-swift-lm-linux patch](evidence/mlx-cuda-dgx-spark-2026-10-12/results/mlx-swift-lm-linux-ea37029.patch.gz). With it, a greedy-generate tool on `MLXLLM` ([swift-generate](evidence/mlx-cuda-dgx-spark-2026-10-12/scripts/swift-generate/)) runs Qwen3-8B 4-bit (MLX affine g64) on the GB10.

**provider-swift on Linux (first error per target).** Each target was built alone ([provider-targets.sh](evidence/mlx-cuda-dgx-spark-2026-10-12/scripts/provider-targets.sh)); a failing target shows only its first blocking errors, so deeper errors in `ProviderCore` are not yet visible.

| Target | Linux build | First blocking cause |
|---|---|---|
| `ProviderCoreFoundation` | PASS | — |
| `ProviderMetallibControl` | PASS | — |
| `darkbloom-publish` | PASS | — |
| `DarkbloomFanCore` | PASS | — |
| `ProviderAppAttest` | FAIL | `import DeviceCheck` (App Attest) |
| `DarkbloomFanProtocol`, `DarkbloomFanService`, `DarkbloomFanHelper` | FAIL | `FanIPC.swift`: Objective-C interoperability (XPC) |
| `ProviderCore`, `ProviderBenchmark`, `darkbloom`, `DarkbloomEnclaveCLI` | FAIL | `import DeviceCheck` via `ProviderAppAttest`; `sodium.h` not found (`swift-sodium` expects a system libsodium) |

Receipts: [provider-targets SUMMARY](evidence/mlx-cuda-dgx-spark-2026-10-12/results/provider-targets-SUMMARY.tsv.gz), [mlx-swift build log](evidence/mlx-cuda-dgx-spark-2026-10-12/results/build-mlx-swift.log.gz), [toolchain](evidence/mlx-cuda-dgx-spark-2026-10-12/results/swift-toolchain.txt.gz).

**Linux provider path (source review).** mlx-swift vendored fork lists Linux/CUDA in SwiftPM; **`MLXFast.metalKernel`** is a fatal stub on Linux and mlx-swift-lm contains **68** `metalKernel` call sites in 39 files ([inventory](evidence/mlx-cuda-dgx-spark-2026-10-12/results/source-counts/mlx-swift-lm-metalKernel-sites.txt.gz))—GatedDelta hits the stub before ops fallback. In the coordinator copy retained in this repository at `7796fe2`, the default trust floor requires Apple attestation hardware claims; private-text routing requires backend **`mlx-swift`** exactly. Capability value **`apple_m5`** (not `apple-m5`) gates one promotion path. A separate **`mlx-cuda`** backend and lower trust policy would be needed for an honest Linux box—**not built or tested** in this measurement set.

## Evidence And Reproduction

Scripts and pinned outputs live under [evidence/mlx-cuda-dgx-spark-2026-10-12/](evidence/mlx-cuda-dgx-spark-2026-10-12/) ([manifest.json](evidence/mlx-cuda-dgx-spark-2026-10-12/manifest.json)).

| Evidence | SHA-256 |
|---|---|
| `results/run2/bench-q3-bf16.json.gz` | `6d260a363b797f54a68c20604aa28087ec933f8c6942d132061b1152f0b01bc8` |
| `results/run2/bench-q3-q4.json.gz` | `ee3a3e22741b596bfa60c282792f030af663dac4459b431ce7e61acd49349075` |
| `results/run2/bench-q35-bf16.json.gz` | `bfa39da5b6ef7c9f4c6de35cf1bda96724368f8b829415717e75782adc21d3e6` |
| `results/run2/bench-q35-q4.json.gz` | `1c03a08215abebd06cc3b0be4e19978228dc644ad391cf002627a6f61c700b2b` |
| `results/run4/bench-q35-bf16-gdncuda.json.gz` | `7007146e73e902452a1ad5ccaf5f10e85247b2778f0bb3e12d2e6d886a723e1f` |
| `results/run4/bench-q35-q4-gdncuda.json.gz` | `0f2c40edcdcf4f17b8fefc5f86a1071a269826ead812e4dd9cc2ef9e340e4f14` |
| `results/run4/test-gdn-cuda.txt.gz` | `62cc6f01dd09f2d55905c0897337f8a4e106f559f01f6d0ec8fba519208cf238` |
| `results/run5/bench-q3-q4-base.json.gz` | `36e7ef0c95f42d37dcba6a5f56b5c9df95261a9725f3d0a692e6c37753669c8d` |
| `results/run5/bench-q3-q4-pip.json.gz` | `add7942e4904d267a730d3e1f6882cd7a60ab38148399c8d94290eb26a40e02f` |
| `results/run5/bench-q3-q4-qmm.json.gz` | `f5b182db5cc91fc71c244d4067a218432172abb5e87f4ef12f53fe5ac322700e` |
| `results/run5/bench-q35-q4-gdncuda-base.json.gz` | `ec48d14d2046fb324978fde4d117ed9d25e1e870a70b0ad723e4e2ba338f1764` |
| `results/run5/bench-q35-q4-gdncuda-pip.json.gz` | `84d38389b6ce4d2201eaaba21c12dbcd96069771e0672057ee44de2b4c70c571` |
| `results/run5/bench-q35-q4-gdncuda-qmm.json.gz` | `d2f7b91e76f59ef9b79745483890a9ca50e39598ef1c5f11fec60f9e51c4ba2e` |
| `results/tf-m5-metal-4bit.json.gz` | `4f962e8b207a5c32017a537946564bcf56921966aecbb1f4199c3d3ac726e5e0` |
| `results/tf-spark-a-cuda-4bit.json.gz` | `b6cbd617e62e5f14ff7c9a2efc3528756f8b30c45b110769c8ba30fd4e9fbaff` |

| Step | Command (from evidence folder; models and venv on the bench host) |
|---|---|
| Cross-platform trace | `python3 scripts/mlx_trace.py run --model MODEL --prompts scripts/trace-prompts.jsonl --out trace.jsonl` |
| Teacher forcing | `python3 scripts/tf_logprobs.py --model MODEL --trace TRACE.jsonl --out tf.json` |
| Speed compare | `python3 scripts/bench_compare.py --mlx-model PATH --gguf PATH --llama-server PATH --prompt-file PROMPT --out result.json --mlx-patch gdn_cuda` (patch optional) |
| Gdn unit test | `python3 scripts/test_gdn_cuda.py` |
| QMM micro | `python3 scripts/qmm_micro.py` (see `--help`) |
| MLX source wheels | `sh scripts/build-mlx-src.sh base` and `sh scripts/build-mlx-src.sh qmm` on GB10-class Linux |

See [evidence README](evidence/mlx-cuda-dgx-spark-2026-10-12/README.md) for folder layout and gzip naming.

**mlx-serve reference (not executed here).** At commit **`6e15b046`**, the Zig server builds MLX **0.32.3** from source via `scripts/build-mlx-linux.sh` (three `git apply` steps for `patches/mlx-cuda-*.patch` when `MLX_BACKEND=cuda`, lines 103–122) and registers one custom gated-delta recurrence through `gdnCudaSource` and `getGdnCudaKernel` in `src/transformer.zig`. On CUDA, the availability check in `src/mlx.zig:91-138` (`cudaAvailable` / `metalKernelsAvailable`) sends **109** `mlx_fast_metal_kernel_new` call sites in 35 files ([inventory](evidence/mlx-cuda-dgx-spark-2026-10-12/results/source-counts/mlx-serve-metal-kernel-sites.txt.gz)) to the MLX-op path; MTP speculative decode is off by default on CUDA (`src/main.zig:1054-1056`).

## Validation And Limits

Numbers were checked against the evidence files by two independent reviews.

- **Single accelerator class:** GB10 sm_121 only; no multi-GPU or fleet measurements.
- **Not shown:** Darkbloom coordinator HTTP serving, provider registration, App Attest, or private-text routing on Linux.
- **Timing:** client-side tok/s only; not TTFT under coordinator deadlines or batching.
- **Statistics:** **5** reps per cell; dense decode ratios can shift **~5** points between sessions—insufficient to claim **1–2 %** effects on dense models.
- **Quantization:** MLX 4-bit vs GGUF `Q4_0` diverge early in greedy cross-engine checks; speed ratios still compare end-to-end wall time.
- **Kernel / patch scope:** gdn_cuda local test only; qmm patch did not improve this harness; mlx-serve upstream remains read-only reference.
- **Swift/CUDA:** mlx-swift builds only with the six local workarounds above, and with an unsupported clang/CUDA pair; the full provider stack does not build on Linux.
