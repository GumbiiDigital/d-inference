# MLX on AMD ROCm: Strix Halo gfx1151 bring-up and speed baseline

> Last updated: 2026-10-12

**MLX runs on an AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151) through the experimental ROCm backend of PR ml-explore/mlx#2300.** Unmodified mlx-lm 0.32.0 loads Gemma 4 26B A4B 4-bit and Qwen3-8B 4-bit and generates on the GPU. For Gemma 4, GPU greedy output equals the CPU backend of the same build for all 128 tokens of the test prompt. Two GPU runs repeat exactly. **It also equals an Apple M5 Max (MLX Metal) reference for all 128 tokens**, free-running and teacher-forced; the one exact tie (step 81) resolves to the same token on both. Speed is the open problem. On the same box, MLX-ROCm decodes Gemma 4 at **41.5 tok/s against 57.1 (llama.cpp HIP) and 59.7 (Vulkan)**, about **−27 %** against HIP; prefill is about **−81 %**. Two measured changes raise Qwen3-8B decode from 29.3 tok/s to an extrapolated ~36.8. Limits: one box, one day, one prompt per token check, different benchmark tools and different weight files for MLX and llama.cpp. Darkbloom's Swift provider does not build on Linux today.

## Results

### Token Agreement

| Model | Check | Steps | Result |
|---|---|---:|---|
| Gemma 4 26B A4B 4-bit | Halo GPU vs Halo CPU, free-running greedy | 128 | identical |
| Gemma 4 26B A4B 4-bit | Halo GPU, two runs | 128 | identical |
| Gemma 4 26B A4B 4-bit | Halo GPU vs M5 Metal, free-running greedy | 128 | identical |
| Gemma 4 26B A4B 4-bit | Halo GPU teacher-forced on M5 Metal tokens | 128 | 128 argmax matches; step 81 is an exact top-2 tie on both devices |
| Qwen3-0.6B 4-bit | Halo GPU vs Halo CPU, free-running greedy, raw prompt | 128 | identical |

### Speed Against llama.cpp On The Same Box

Single stream, prompt 512, decode 128. Values are means of 2 rounds unless marked †.

| Model | Engine | Weights | Prefill tok/s | Decode tok/s |
|---|---|---|---:|---:|
| Gemma 4 26B A4B | MLX-ROCm | MLX 4-bit, 15.3 GB | 241 | 41.5 |
| Gemma 4 26B A4B | llama.cpp HIP | GGUF QAT Q4_0, 14.4 GB | 1,285 | 60.5 (57.1 at depth 512) |
| Gemma 4 26B A4B | llama.cpp Vulkan | GGUF QAT Q4_0, 14.4 GB | 1,330 | 64.7 (59.7 at depth 512) |
| Qwen3-8B | MLX-ROCm | MLX 4-bit, 4.61 GB | 425 | 29.2 |
| Qwen3-8B | llama.cpp HIP | Q4_0, 4.77 GB | 1,156 | 44.9 (44.3 at depth 512 †) |
| Qwen3-8B | llama.cpp Vulkan | Q4_0, 4.77 GB | 1,114 | 44.8 (43.7 at depth 512 †) |
| Qwen3-8B | llama.cpp HIP | Q4_K_M, 5.03 GB | 1,062 | 41.5 |
| Qwen3-8B | llama.cpp Vulkan | Q4_K_M, 5.03 GB | 954 | 42.5 |

† one round of 5 repetitions.

Decode at matched depth: Gemma 4 −27 % against HIP and −30 % against Vulkan; Qwen3-8B −34 % against HIP Q4_0. Prefill: −81 % for Gemma 4 and −63 % for Qwen3-8B, both against HIP. Qwen3-0.6B decodes at 215 tok/s on MLX against 333 (Vulkan) and 285 (HIP).

### Decode Profile (Qwen3-8B)

| Kernel group | ms per token | Share |
|---|---:|---:|
| `qmv_wide_kernel` (4-bit matrix-vector), 253 launches | 24.7 | 92.5 % |
| All other kernels | ~2.0 | ~7.5 % |
| Total GPU kernel time | 26.8 | 100 % |

| Projection | Shape N×K | Median µs | Floor µs at 238 GB/s | Floor / time |
|---|---|---:|---:|---:|
| lm_head | 151936×4096 | 1,512 | 1,471 | 97 % |
| gate, up | 12288×4096 | 155, 151 | 119 | 77 %, 79 % |
| down | 4096×12288 | 154 | 119 | 77 % |
| q, o | 4096×4096 | 66 | 40 | 60 % |
| k, v | 1024×4096 | 25 | 10 | 39 % |

The qmv floor is 17.9 ms per token, so the loss is 6.8 ms. The small k/v and q/o matrices lose about 3.0 ms. The large gate/up and down matrices lose about 3.7 ms.

### Decode Changes (Qwen3-8B)

| Change | Kind | Decode | Tokens |
|---|---|---|---|
| `HSA_ENABLE_INTERRUPT=0` | environment only | 29.3 → 34.2 tok/s (+16.7 %), p512/g128 | identical (greedy 128) |
| Fused q+k+v and gate+up projections | model-level prototype | 35.3 → 38.0 tok/s (+7.7 %), 8-token prompt, with the line above | identical |
| Both, at depth 512 | extrapolation 34.2 × 38.0 / 35.3 | ~36.8 tok/s | — |
| `qmv_wide2` kernel variants | MLX fork patch | no gain | bit-identical on 126 shapes |

The interrupt switch does not change llama.cpp HIP (44.9 tok/s with and without). On Gemma 4 it adds about 2 % (41.5 → 42.2). `MLX_EVENT_BLOCKING=1` is 9 % slower. `MLX_QMV_IDOT`, `MLX_ROCM_FINEGRAINED=0` and `MLX_GRAPH_DECODE=0` change nothing measurable. Fusion helps because each `qmv_wide_kernel` launch costs more than its memory traffic: three q/k/v launches take 125 µs, one fused launch 92 µs. The cause of that cost is not proven. The `qmv_wide2` variants add vector x loads, 1–2 output columns per warp and 1–4 loads in flight. With one column per warp they stay within −3 % to +10 % of the existing kernel per call; with two columns they are 16–90 % slower.

### Findings

- **Float32 quantized matmul is wrong on the GPU.** `mx.quantized_matmul` (4-bit, group 64) with float32 activations and scales has a relative error above 5 % in 51 of 64 tested shapes against an exact float32 reference. Some outputs are all zeros. No bfloat16 or float16 case exceeds 5 % in the same sweep. The 4-bit models above run in bfloat16.
- **Gemma 4 IT needs its chat template** in a raw `generate_step` loop. Without it, a 16-token CPU run repeats " the of".
- **The profiler changes decode speed.** A profiled Qwen3-8B run decodes at about 35 tok/s, the same command without it at about 30. In the unprofiled run the host thread is at about 100 % CPU; perf samples include `yield_task_fair` 6.9 %, `__sched_yield` 6.0 % and `HipEvent::wait` 5.0 %.

## Method And Gates

The token reference is a greedy run on an Apple M5 Max (MLX 0.32.3, Metal) with `tokref.py`. The Halo run feeds the reference prompt and then each reference token through the decode path (`teacherforce.py`). At each step it compares its own argmax with the reference token. A different argmax counts as a tie only when the reference token has exactly the top logprob on the Halo. Every other difference is a mismatch. The self-check on the Halo's own tokens gives 128 of 128. A copy with the token at step 40 replaced reports exactly that step as a mismatch. The Gemma 4 prompt uses the model's chat template.

MLX speed uses `mlx_lm.benchmark` (random prompt tokens; decode follows the 512-token prompt; 5 trials per round). llama.cpp uses `llama-bench` with `-ngl 999 -fa 1` at depth 0 and depth 512 (5 repetitions per round). The tools time differently, so ratios are approximate. The weights also differ: the MLX files are community 4-bit conversions, the Gemma 4 GGUF is Google's QAT Q4_0 file, and the Qwen3 Q4_0 GGUF is requantized from the official Q8_0 file.

The profile uses `rocprofv3 --kernel-trace`. Decode cost per kernel is (run with 129 tokens − run with 1 token) / 128. Per-projection medians come from the last 100 tokens of the 129-token trace. Floor bytes are N×K/2 + N×K/64×4 (4-bit weights plus a bf16 scale and bias per group of 64). The floor uses 238 GB/s, the best GPU read of `gpubw.hip` on this box.

Every benchmark ran under one exclusive GPU lock with the box otherwise idle. Unless a row says otherwise, results come from the pinned build. The `qmv_wide2` rows come from build `4e4be447`, which is the pin plus the two patches in the evidence folder; its default path is unchanged.

## Frozen Provenance

| Item | Value |
|---|---|
| Host | AMD Ryzen AI Max+ 395, Radeon 8060S (gfx1151, 40 compute units), LPDDR5 at 8000 MT/s, 125 GiB visible to Linux |
| GPU read bandwidth | 238.4 GB/s (`gpubw.hip`, 1 GiB device buffers) |
| ROCm | `amd-smi` reports ROCm version 10.0.0 |
| MLX | `NripeshN/mlx` branch `rocm-support` at `75915908dfe5028335d318b10340313744fd3a8d` (commit date 2026-07-19; the source of PR ml-explore/mlx#2300); version `0.32.0.dev20261012+75915908` |
| MLX build | `pip install .` with `-DMLX_BUILD_ROCM=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151`; needs `libopenblas-dev liblapack-dev liblapacke-dev` |
| mlx-lm | 0.32.0 wheel, sha256 `9a0f3843135fd88d36ad791f0d82210cdc4130c5c1cb5416f5f0c4b56a135778`, installed `--no-deps` |
| llama.cpp | build `10a60cf30`, Vulkan and HIP |
| Gemma 4 MLX | `mlx-community/gemma-4-26b-a4b-it-4bit` rev `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`; 4-bit affine, group 64, router projections 8-bit |
| Gemma 4 GGUF | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` rev `d1c082be9cf3c8a514acf63b8761f4b41935842e`, `gemma-4-26B_q4_0-it.gguf`, sha256 `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` |
| Qwen3 MLX | `mlx-community/Qwen3-8B-4bit` rev `545dc4251c05440727734bcd94334791f6ab0192`; `mlx-community/Qwen3-0.6B-4bit` rev `73e3e38d981303bc594367cd910ea6eb48349da8` |
| Qwen3 GGUF | `Qwen/Qwen3-8B-GGUF` rev `7c41481f57cb95916b40956ab2f0b139b296d974` (Q8_0, Q4_K_M; Q4_0 made with `llama-quantize --allow-requantize`) |

The listed model file hashes and sizes are in `results/host-and-models/receipts.txt`.

## Linux Build Gaps

Checked at d-inference `4aa44a1f9` with `libs/mlx-swift` `6923a80` (vendored MLX `Layr-Labs/mlx` `cb77239`) and `libs/mlx-swift-lm` `53d96470`:

| Area | State | What a ROCm port needs |
|---|---|---|
| `libs/mlx-swift/Package.swift` | on Linux, CUDA unless `SPM_CUDA=0`, which selects CPU; no ROCm mode | a ROCm mode with a HIP build step, like the CUDA build plugin |
| Vendored MLX (`libs/mlx-swift/Source/Cmlx/mlx`) | backends `cpu`, `cuda`, `metal`; no `rocm` | the ROCm backend of PR ml-explore/mlx#2300, merged with the Layr-Labs patches |
| `libs/mlx-swift/Source/MLX/MLXFastKernel.swift` | without Metal the kernel API calls `fatalError("MLXFastKernel is not available without Metal")` | HIP kernels or generic MLX paths |
| `libs/mlx-swift-lm` | `Package.swift` lists Apple platforms only; 42 Swift files under `Libraries/` call `metalKernel` | the same, for the kernels on each served model's path |
| `provider-swift/Package.swift` | `platforms: [.macOS(.v14)]`; App Attest, Secure Enclave and IOKit targets | Linux builds of the serving targets with those parts off |
| `GPUEnforcement.requireMetal()` (`provider-swift/Sources/ProviderCore/Inference/Engine/GPUEnforcement.swift`) | `StartCommand.run` calls it through `prepareServeRuntime` before the `--local` branch | a non-Metal GPU check |
| Coordinator trust | `Registry.MinTrustLevel` defaults to `TrustHardware` (`coordinator/registry/registry.go`); an owner's self-routed request relaxes it to `TrustNone` (`coordinator/registry/request_traits.go`) | a product decision for non-Apple providers, out of scope here |

A Linux ROCm provider is therefore a port, not a build flag. A lower-risk first step for lab use is a plain MLX-ROCm server (Python mlx-lm or C++) outside the Swift provider.

## Evidence And Reproduction

Everything is in [`evidence/mlx-rocm-strix-halo-2026-10-12/`](evidence/mlx-rocm-strix-halo-2026-10-12/): `scripts/`, `results/`, `patches/` (the `qmv_wide2` negative result) and `manifest.json`. The manifest lists the sha256 and size of every file except `manifest.json` and `make_manifest.py`; `python3 make_manifest.py --check` verifies them. The [evidence README](evidence/mlx-rocm-strix-halo-2026-10-12/README.md) lists each command.

Key result files (paths relative to the evidence folder):

| File | SHA-256 |
|---|---|
| `results/gemma4-26b/tokref-gpu-pinned.json` | `53135ca54fbe0e0b3e4172bc3c4e303c402151cc9faf112501a1bdfc564a6ca4` |
| `results/gemma4-26b/gemma4-gpu-pinned-r2.json` | `48abe40b018cbd8ed69179895af141bd552425f2a55e0c1bb4eca06b9e4d8fee` |
| `results/gemma4-26b/gemma4-cpu-pinned.json` | `743b4bc00921de84229802acf4c58ff6f6e06182901c18e57b12421f5d9e70a5` |
| `results/gemma4-26b/gemma4-m5-reference.json` | `56f336253c0888f91b30864c7d70fd339ebcff82582c32cf8828101d05501903` |
| `results/gemma4-26b/tf-vs-m5.json` | `70b9b3f6ca865fb500a9531b289528d4e2488648122af4b10ec217a9180d3583` |
| `results/gemma4-26b/tf-self.json` | `57deb325405c881ae406d81d5f06a6482703ceff5de4259e85af3a308cf72e79` |
| `results/gemma4-26b/hip-r1.json` | `b4588e490ea02dcbc6f2110a498c5c8c6f970b31d60d2cfcd380e81ed23ab290` |
| `results/gemma4-26b/vk-r1.json` | `b71a1ba2d0e5d9b21efe9e2dfcbe840c1861bad8268dd4b3a5c0b4438da5d6a8` |
| `results/gemma4-26b/mlx-default-r1.txt` | `797085c60b88f24e9aef53a69e4799e0510a4062a573a20796d8aee95170e127` |
| `results/qwen3-8b-profile/prof_reduce-output.txt` | `10a8bb5efdf391b73bdc20c774daff3acc473b545f3703ec5d716c07a3299e9e` |
| `results/qmm-dtype-check/qmmcheck-output.txt` | `cd2c773cac087d77d65f16c939d1bc6dd704afe71581d661ce0b1a52c5b674e5` |
| `results/bandwidth/gpubw-output.txt` | `95d46421cf9e4f333dcdf7770c72d7b350acda49ec3ded48c879ecc2a23e163e` |
| `results/host-and-models/receipts.txt` | `2461ff7a83782761f5a1e3b212b5de06ee834581aa921c5033f7e9869aeca33b` |

## Validation And Limits

- One box, one day, one prompt per token check, one MLX fork commit (plus the patched `4e4be447` build for the `qmv_wide2` rows).
- MLX and llama.cpp use different benchmark tools and different weight files; ratios are approximate.
- The fused-projection gain uses an 8-token prompt. The ~36.8 tok/s figure is an extrapolation, not a measurement.
- The M5 reference is a separate run on another machine (MLX 0.32.3, Metal).
- The cause of the per-launch cost in `qmv_wide_kernel` is not proven. Prefill was measured but not profiled.
- The float32 sweep covers 64 shapes per dtype at group 64, 4-bit only.
