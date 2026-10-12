#!/usr/bin/env python3
import argparse
import json
import os
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

from fuse_qwen3 import fuse

PROMPT = "The history of the printing press begins in"


def _run_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if "HSA_ENABLE_INTERRUPT" in os.environ:
        env["HSA_ENABLE_INTERRUPT"] = os.environ["HSA_ENABLE_INTERRUPT"]
    for key, val in os.environ.items():
        if key.startswith("MLX_"):
            env[key] = val
    return env


def greedy_decode(
    model, ids, max_tokens: int
) -> tuple[list[int], float, int]:
    prompt = mx.array(ids)
    sampler = lambda logits: mx.argmax(logits, axis=-1)
    gen: list[int] = list(ids)
    start = None
    count = 0
    for token, _ in generate_step(
        prompt, model, max_tokens=max_tokens, sampler=sampler
    ):
        gen.append(int(token))
        count += 1
        if start is None:
            start = time.perf_counter()
    assert count >= 2, f"expected at least 2 generated tokens, got {count}"
    elapsed = time.perf_counter() - start
    decode_tok_s = (count - 1) / elapsed if elapsed > 0 else float("inf")
    return gen, decode_tok_s, count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("--fuse", action="store_true")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--out", metavar="file.json")
    args = p.parse_args()

    model, tokenizer = load(args.model_path)
    fused = 0
    if args.fuse:
        fused = fuse(model)

    ids = tokenizer.encode(PROMPT)
    greedy_decode(model, ids, args.tokens)
    gen_ids, decode_tok_s, n_generated = greedy_decode(model, ids, args.tokens)

    result = {
        "fused": fused,
        "gen_ids": gen_ids,
        "decode_tok_s": decode_tok_s,
        "n_generated": n_generated,
        "prompt_len": len(ids),
        "env": _run_env(),
    }
    text = json.dumps(result)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
