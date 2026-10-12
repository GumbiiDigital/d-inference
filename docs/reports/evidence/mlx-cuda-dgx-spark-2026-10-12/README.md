# MLX-CUDA DGX Spark evidence (2026-10-12)

> Last updated: 2026-10-12

## Layout

| Path | Contents |
|---|---|
| `scripts/` | Measurement scripts (`mlx_trace.py`, `bench_compare.py`, `tf_logprobs.py`, `gdn_cuda.py`, `test_gdn_cuda.py`, `qmm_micro.py`), box build helpers, `trace-prompts.jsonl`, Swift smoke/generate packages (sources only). |
| `results/` | Gzipped (`gzip -n`) JSON, JSONL, logs, and env pin files; hostnames scrubbed to spark-a / spark-b / m5-max and home paths to `$HOME`. |
| `manifest.json` | Schema 1 file list with sha256 (and uncompressed hashes for `.gz`). |

## Reproduce (on GB10-class Linux with models, venv, and llama.cpp built)

```bash
# Greedy trace (no chat template, EOS ignored)
python3 scripts/mlx_trace.py run --model MODEL_DIR --prompts scripts/trace-prompts.jsonl \
  --out results/trace.jsonl --max-tokens 64 --label LABEL

# Teacher-forced logprobs vs a trace
python3 scripts/tf_logprobs.py --model MODEL_DIR --trace TRACE.jsonl --out results/tf.json

# MLX vs llama-server (interleaved, 5 reps); optional gated-delta patch
python3 scripts/bench_compare.py --mlx-model MLX_DIR --gguf GGUF_PATH \
  --llama-server PATH/to/llama-server --prompt-file data/prompt-text.txt \
  --out results/bench.json --prefill-lens 512,2048 --decode-tokens 128 --reps 5 \
  --llama-extra-args "-fa on" --gpu-temp-max 60 --gpu-util-max 10 \
  --mlx-patch gdn_cuda

# Gated-delta kernel unit test (CUDA)
python3 scripts/test_gdn_cuda.py

# Quantized matmul micro-benchmark
python3 scripts/qmm_micro.py --help

# Build MLX 0.32.3 wheels from source (base vs qmm patch)
sh scripts/build-mlx-src.sh base
sh scripts/build-mlx-src.sh qmm
```

Compare traces: `python3 scripts/mlx_trace.py compare A.jsonl B.jsonl`

## Swift on Linux (DGX Spark B)

- `scripts/build-mlx-swift.sh` builds `scripts/swift-smoke/` (one GPU matmul) against `libs/mlx-swift` with the workarounds listed in the report.
- `scripts/run-swift-generate.sh` builds `scripts/swift-generate/` (greedy generate on `MLXLLM`), runs it and the Python trace on the same model, then `mlx_trace.py compare`.
- `scripts/provider-targets.sh` builds each `provider-swift` target alone and writes `results/provider-targets-SUMMARY.tsv`.
- `results/mlx-swift-lm-linux-ea37029.patch.gz`: the local platform-guard patch for `libs/mlx-swift-lm` at `3fd4944`. `results/mlx-swift-nvrtc-include.patch.gz`: the local NVRTC include-path change in the MLX core.

Prompt input: the benchmark prompt was the text of llama.cpp `README.md` followed by `docs/build.md` at tag `b11573` (commit `0f701c8`), tokenised with the MLX tokenizer. That file is not copied here; the exact token ids used for P=512 and P=2048 are retained in every bench JSON (`prompt_ids`). To replay without the text, extract them into a JSON object `{"512": [...], "2048": [...]}` and pass `--prompt-ids FILE` instead of `--prompt-file`.
