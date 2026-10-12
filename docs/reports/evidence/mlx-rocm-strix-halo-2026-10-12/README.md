# MLX on AMD ROCm (Strix Halo, gfx1151) — evidence bundle

> Last updated: 2026-10-12

Frozen measurement artifacts for the report dated 2026-10-12. Numbers and token ids in `results/` are as captured; this tree does not re-run benchmarks.

## Contents

- `scripts/` — build and benchmark helpers used on the measurement host.
- `results/gemma4-26b/` — Gemma 4 26B A4B MLX vs llama.cpp (Vulkan/HIP), token reference, teacher-forced self-check, cross-build tokref JSON.
- `results/gemma4-26b/gemma4-gpu-pinned-r2.json`, `gemma4-cpu-pinned.json`, `build-pinned-version.txt` — second GPU tokref and CPU tokref on pinned build `75915908` (`scripts/tokref.py`, `gpu`/`cpu`).
- `results/gemma4-26b/tf-tamper-ref.json`, `tf-tamper-result.json` — copy of `tokref-gpu-pinned.json` with `gen_ids[40]` set to that step's second-best token; `teacherforce.py` on GPU reports step 40 as the only mismatch.
- `results/gemma4-26b/raw-prompt-cpu16.json` — 16 tokens on CPU with a raw prompt (no chat template), from an earlier `tokref.py` that used `tok.encode(prompt)` and prompt `The history of the printing press begins in`; build `4e4be447`.
- `results/qmv-wide2/` — `qmvbench-wide2-{old,11,12,14,21,22,24}.txt` from `scripts/qmvbench.py` at each `MLX_QMV_WIDE2` with `HSA_ENABLE_INTERRUPT=0`; `wide2check-compare.txt` from `scripts/wide2check.py` (per setting) then `scripts/wide2compare.py`; `build-wide2.txt` records build `4e4be447` (pin + `patches/`).
- `results/bandwidth/gpubw-output.txt` — output of `scripts/gpubw.hip`.
- `results/qwen3-0.6b-tokens/diag-gpu.json`, `diag-cpu.json` — greedy 128 tokens on Qwen3-0.6B-4bit (raw prompt), top-2 logprob gaps and warm tok/s (`scripts/diag.py`).
- `results/qwen3-8b-env-sweep/` — MLX-ROCm environment-switch decode sweep (greedy 128 tokens, text parity vs base).
- `results/qwen3-8b-fusion/` — Prototype fused QKV / gate-up projection runs (`run_fuse.py`).
- `results/qwen3-8b-profile/` — rocprofv3 kernel trace for decode length 1 vs 129 (`kernel_trace_decode1.csv.gz`, `kernel_trace_decode129.csv.gz`), derived kernel stats CSV, and `prof_reduce-output.txt` from `scripts/prof_reduce.py`.
- `results/qwen3-8b-speed/` — Qwen3 8B and 0.6B speed: llama.cpp Vulkan/HIP from `scripts/step3.sh`; MLX-ROCm from `scripts/step3-mlx.sh` (snapshot paths). `step3.sh` MLX lines failed offline and were rerun via `step3-mlx.sh`. Also interrupt A/B, depth-512 llama-bench, pip freeze, amd-smi version.
- `patches/` — Measured qmv_wide2 experiments (negative result), not upstream proposals.

## Paths

On the measurement host, shell scripts default to:

- `WORK` — MLX-ROCm working tree with venv and run outputs (default `$HOME/mlxrocm`).
- `MODELS` — local GGUF files (default `$HOME/models`).
- `LLAMA_BUILD` — llama.cpp builds with `up-vk/` and `up-hip/` (default `$HOME/build`).
- Hugging Face hub cache — `$HF_HOME/hub` or `$HOME/.cache/huggingface/hub` when `HF_HOME` is unset.

Frozen files under `results/` replace those locations with placeholders: `<work>`, `<models>`, `<llama-build>`, `<hf-cache>`, and `<home>` (any remaining home-relative path).

## Manifest

`manifest.json` lists every file in this bundle except `manifest.json` itself and `make_manifest.py`. Regenerate with `python3 make_manifest.py`; verify with `python3 make_manifest.py --check`.

## Environment

Ryzen AI Max+ 395 / Radeon 8060S gfx1151, 128 GB unified memory.

MLX source: [NripeshN/mlx](https://github.com/NripeshN/mlx) branch `rocm-support`, commit `75915908dfe5028335d318b10340313744fd3a8d` (PR [ml-explore/mlx#2300](https://github.com/ml-explore/mlx/pull/2300)), built with `scripts/build.sh`.

mlx-lm 0.32.0 wheel sha256 `9a0f3843135fd88d36ad791f0d82210cdc4130c5c1cb5416f5f0c4b56a135778`.

llama.cpp build `10a60cf30`.

amd-smi reports ROCm version 10.0.0 (see `results/qwen3-8b-speed/amd-smi-version.txt`).

## Models

- mlx-community/Qwen3-8B-4bit rev `545dc4251c05440727734bcd94334791f6ab0192` — `model.safetensors` 4607835174 bytes, sha256 `f2d29621aab300336ad645567ff38c42aac755513006ef4e8a579cf7ef5256d8`
- mlx-community/Qwen3-0.6B-4bit rev `73e3e38d981303bc594367cd910ea6eb48349da8`
- mlx-community/gemma-4-26b-a4b-it-4bit rev `0d77464eeb233a2da68ebf9d7dc4edaac7db956d` — three safetensors: 5320218487 bytes sha256 `683f420ee09550b8027bf0335c4202e196b37eb520b60a4bcbe8690b9e388c07`; 5363328422 bytes sha256 `feab2873c2976fb7ed666f8462549564a29409364388e0c8f6b015729d165e15`; 4657658867 bytes sha256 `bc607486deb1de5bc7e459932fda5c776a4f816b368feaa661bab8cfbb47e567`
- Qwen/Qwen3-8B-GGUF rev `7c41481f57cb95916b40956ab2f0b139b296d974` — Q8_0 and Q4_K_M; Q4_0 made with `llama-quantize --allow-requantize` from Q8_0
- Qwen/Qwen3-0.6B-GGUF rev `23749fefcc72300e3a2ad315e1317431b06b590a` — Q8_0 → Q4_0 the same way
- google/gemma-4-26B-A4B-it-qat-q4_0-gguf rev `d1c082be9cf3c8a514acf63b8761f4b41935842e` — file `gemma-4-26B_q4_0-it.gguf`, 14439363584 bytes, sha256 `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` (Google QAT Q4_0 weights; **not** the same weights as the mlx-community 4-bit MLX conversion above)

## Reproduce

Requires ROCm, an MLX-ROCm venv under `WORK`, llama.cpp builds under `$LLAMA_BUILD/up-vk` and `$LLAMA_BUILD/up-hip`, and offline Hugging Face caches for the model revisions above. Run every command from **this evidence folder** (paths below are relative to it).

Every benchmark on the measurement host was started under an exclusive GPU lock, for example `flock /tmp/mtplx-gpu-exclusive.lock bash scripts/step3.sh`. The scripts do not take the lock themselves.

- `flock /tmp/mtplx-gpu-exclusive.lock bash scripts/build.sh` — build MLX-ROCm into the active venv; writes `$WORK/build.log` and `$WORK/pip-freeze.txt`
- `flock /tmp/mtplx-gpu-exclusive.lock bash scripts/step3.sh` — two-round Vulkan vs HIP llama-bench on Qwen3 8B/0.6B (and attempted MLX); writes `$WORK/step3/`
- `flock /tmp/mtplx-gpu-exclusive.lock bash scripts/step3-mlx.sh` — MLX-only Qwen3 8B/0.6B benchmarks using pinned snapshot paths; writes `$WORK/step3/mlx-*-r*.txt`
- `flock /tmp/mtplx-gpu-exclusive.lock bash scripts/prof.sh` — rocprofv3 kernel trace for `-m 1` and `-m 129` generate; writes `$WORK/prof/` (gzip CSV traces copied to `results/qwen3-8b-profile/kernel_trace_decode{1,129}.csv.gz`)
- `python3 scripts/prof_reduce.py` — stdlib reduction of those two traces → `results/qwen3-8b-profile/prof_reduce-output.txt`
- `flock /tmp/mtplx-gpu-exclusive.lock bash scripts/sweep.sh` — env-switch decode sweep (base, blocking, coarse, both, idot, nograph); writes `$WORK/sweep/`
- `flock /tmp/mtplx-gpu-exclusive.lock bash scripts/sweep2.sh` — second env sweep (serial3, nointr, kernarg, profcheck); same output dir
- `flock /tmp/mtplx-gpu-exclusive.lock bash scripts/gemma-bench.sh` — Gemma 4 26B MLX (default and no-interrupt) vs llama-bench; writes `$WORK/gemma/`
- Qwen3-8B interrupt A/B (`results/qwen3-8b-speed/interrupt-*`): with `$M` = Qwen3-8B snapshot path, two rounds each of `HSA_ENABLE_INTERRUPT=0 python -m mlx_lm.benchmark --model $M -p 512 -g 128 -n 5` and the same without the env var → `interrupt-nointr-8b-r{1,2}.txt` / `interrupt-base-8b-r{1,2}.txt`; HIP no-interrupt llama-bench → `interrupt-hip-nointr.json`: `HSA_ENABLE_INTERRUPT=0 $LLAMA_BUILD/up-hip/bin/llama-bench -m $MODELS/Qwen_Qwen3-8B-GGUF/Qwen3-8B-Q4_0.gguf -ngl 999 -fa 1 -p 512 -n 128 -d 0,512 -r 5 -o json`
- Depth-512 llama-bench (`results/qwen3-8b-speed/*-8b-Q4_0-d512.json`, one round): `$LLAMA_BUILD/up-vk/bin/llama-bench` or `up-hip/bin/llama-bench -m $MODELS/Qwen_Qwen3-8B-GGUF/Qwen3-8B-Q4_0.gguf -ngl 999 -fa 1 -p 0 -n 128 -d 512 -r 5 -o json`
- `python3 scripts/tokref.py MODEL REVISION gpu|cpu OUT.json [N]` — greedy token reference with top-2 logprob gaps
- `python3 scripts/teacherforce.py REF.json gpu|cpu OUT.json` — teacher-forced decode check against a tokref JSON
- `python3 scripts/fusebench.py` — microbench separate vs fused qmv shapes
- `scripts/fuse_qwen3.py` — fuse Qwen3 quantized layers (imported by `run_fuse.py` / `check_fuse.py`; no CLI)
- `python3 scripts/run_fuse.py MODEL_PATH [--fuse] [--tokens 128] [--out file.json]` — greedy decode benchmark with optional fusion
- `python3 scripts/check_fuse.py` — CPU self-check for `concat_qlinear`
- `python3 scripts/qmmcheck.py` — GPU vs CPU quantized_matmul sanity scan
- `python3 scripts/qmvbench.py` — timed qmv per Qwen3-8B projection shape
- `python3 scripts/wide2check.py OUT.npz` — fixed-seed qmv outputs for wide2 A/B; for each `MLX_QMV_WIDE2` setting run wide2check, then `python3 scripts/wide2compare.py OUT_DIR` → `wide2check-compare.txt`
- On MLX pin `75915908` (see `results/gemma4-26b/build-pinned-version.txt`): `python3 scripts/tokref.py mlx-community/gemma-4-26b-a4b-it-4bit 0d77464eeb233a2da68ebf9d7dc4edaac7db956d gpu results/gemma4-26b/gemma4-gpu-pinned-r2.json`; same with `cpu` → `gemma4-cpu-pinned.json`
- Tamper teacher-force: copy `results/gemma4-26b/tokref-gpu-pinned.json` to `tf-tamper-ref.json`, set `gen_ids[40]` to `top2[40][1]`; `python3 scripts/teacherforce.py results/gemma4-26b/tf-tamper-ref.json gpu results/gemma4-26b/tf-tamper-result.json`
- `results/gemma4-26b/raw-prompt-cpu16.json` — captured on build `4e4be447` with a prior `tokref.py` (raw encode, no chat template, 16 tokens, CPU); not reproduced by current `tokref.py`
- Build `4e4be447` with pin + `patches/` (see `results/qmv-wide2/build-wide2.txt`); for each `MLX_QMV_WIDE2` in old/11/12/14/21/22/24: `HSA_ENABLE_INTERRUPT=0 MLX_QMV_WIDE2=<val> python3 scripts/qmvbench.py` → `results/qmv-wide2/qmvbench-wide2-<val>.txt`; `python3 scripts/wide2check.py OUT_DIR/wide2-<val>.npz` for each setting, then `python3 scripts/wide2compare.py OUT_DIR` → `wide2check-compare.txt`
- `hipcc -O3 --offload-arch=gfx1151 -o gpubw scripts/gpubw.hip && ./gpubw` → `results/bandwidth/gpubw-output.txt`
- `python3 scripts/diag.py gpu|cpu OUT.json` — greedy 128 tokens on Qwen3-0.6B-4bit (raw prompt), top-2 gaps and warm tok/s; e.g. `python3 scripts/diag.py gpu results/qwen3-0.6b-tokens/diag-gpu.json`

- `results/gemma4-26b/gemma4-m5-reference.json`: the M5 Max reference (MLX 0.32.3, Metal), made with `scripts/tokref.py` unchanged; the `host` field was removed. Step 81 is its only gap below 0.1 (0.0, an exact tie).
- `results/gemma4-26b/tf-vs-m5.json`: `python scripts/teacherforce.py results/gemma4-26b/gemma4-m5-reference.json gpu OUT.json` on the pinned build.
