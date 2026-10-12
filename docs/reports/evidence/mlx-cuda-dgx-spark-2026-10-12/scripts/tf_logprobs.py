"""Teacher-forced logprobs: feed a fixed token sequence, record per-position logprob of the next token.

Usage:
  python -I tf_logprobs.py --model PATH --trace TRACE.jsonl --out OUT.json [--extend 448]
Takes prompt_ids + tokens of each prompt in TRACE (normally the Metal reference trace), optionally
extends the sequence with --extend copies of its own tokens (cycled) to probe longer positions, and runs:
  prefill: one forward pass over the whole sequence
  decode:  one token at a time with a KV cache
Output per prompt: the next-token logprob and argmax per position, for both paths.
"""
import argparse, importlib.util, json, platform, sys
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache


def install_mlx_patch(name):
    if not name:
        return None
    if name != "gdn_cuda":
        sys.exit(f"error: unknown --mlx-patch {name!r}; use gdn_cuda")
    path = Path(__file__).resolve().parent / "gdn_cuda.py"
    spec = importlib.util.spec_from_file_location("gdn_cuda", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.install()
    return mod


def record_gdn_patch_env(env, gdn_mod):
    status = gdn_mod.qwen3_5_patch_loaded()
    if status is not None:
        env["patched_qwen3_5"] = status
        if not status:
            sys.exit("error: mlx_lm.models.qwen3_5 is loaded but gated_delta_update is not patched")


def lp_rows(logits, targets):
    lp = logits.astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    tgt = mx.take_along_axis(lp, mx.array(targets)[:, None], axis=-1)[:, 0]
    am = mx.argmax(lp, axis=-1)
    mx.eval(tgt, am)
    return tgt.tolist(), am.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extend", type=int, default=0)
    ap.add_argument("--mlx-patch", default=None, help="optional mlx-lm patch (e.g. gdn_cuda)")
    a = ap.parse_args()
    env = {"platform": platform.platform(), "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
           "device_info": mx.device_info() if hasattr(mx, "device_info") else None,
           "model": a.model, "trace": a.trace, "extend": a.extend}
    gdn_mod = install_mlx_patch(a.mlx_patch) if a.mlx_patch else None
    if gdn_mod is not None:
        env["mlx_patch"] = gdn_mod._INSTALL_REPORT or {"mlx_lm.models.gated_delta": True}
    model, _tok = mlx_lm.load(a.model)
    if gdn_mod is not None:
        record_gdn_patch_env(env, gdn_mod)
    out = {"env": env, "prompts": {}}
    for line in open(a.trace):
        r = json.loads(line)
        if "id" not in r:
            continue
        seq = list(r["prompt_ids"]) + list(r["tokens"])
        gen = list(r["tokens"])
        i = 0
        while len(seq) < len(r["prompt_ids"]) + len(r["tokens"]) + a.extend:
            seq.append(gen[i % len(gen)]); i += 1
        x, tgt = seq[:-1], seq[1:]
        # prefill path
        logits = model(mx.array(x)[None], cache=make_prompt_cache(model))[0]
        p_lp, p_am = lp_rows(logits, tgt)
        # decode path
        cache = make_prompt_cache(model)
        rows = []
        for t in x:
            row = model(mx.array([[t]]), cache=cache)[0, -1]
            mx.eval(row)  # evaluate per step, as real decoding does (one huge lazy graph thrashes the CUDA graph cache)
            rows.append(row)
        d_lp, d_am = lp_rows(mx.stack(rows), tgt)
        out["prompts"][r["id"]] = {"n_prompt": len(r["prompt_ids"]), "seq": seq,
                                   "prefill_lp": p_lp, "prefill_argmax": p_am,
                                   "decode_lp": d_lp, "decode_argmax": d_am}
        print(r["id"], len(seq), flush=True)
    if gdn_mod is not None:
        env["mlx_patch_counts"] = gdn_mod.counts()
        print(f"mlx_patch_counts: {json.dumps(env['mlx_patch_counts'])}", file=sys.stderr)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
