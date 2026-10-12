#!/usr/bin/env python3
"""Greedy token trace for a cross-platform MLX match check.

Usage:
  python3 mlx_trace.py run --model PATH --prompts prompts.jsonl --out trace.jsonl \
      [--max-tokens 64] [--label "rev abc123"]
  python3 mlx_trace.py compare A.jsonl B.jsonl

prompts.jsonl: one JSON object per line: {"id": "p1", "text": "..."}.
No chat template is applied. Decoding is greedy (argmax) and does not stop on EOS.
compare exit codes: 0 = all prompts match fully, 1 = a prompt diverges,
2 = prompt_ids differ for a prompt (comparison invalid) or no common prompt ids.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

NEAR_TIE = 0.05  # nats


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_prompts(path):
    prompts = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" not in rec or "text" not in rec:
                sys.exit(f"error: {path}:{n}: each line needs 'id' and 'text'")
            prompts.append(rec)
    if not prompts:
        sys.exit(f"error: no prompts in {path}")
    return prompts


def env_record(mx, mlx_lm, args):
    model_path = Path(args.model)
    shas = {}
    for p in sorted(model_path.rglob("*.safetensors")):
        shas[str(p.relative_to(model_path))] = sha256_file(p)
    info = None
    if hasattr(mx, "device_info"):
        try:
            info = mx.device_info()
        except Exception as e:  # record, do not hide
            info = {"error": repr(e)}
    return {
        "type": "env",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "default_device": str(mx.default_device()),
        "device_info": info,
        "label": args.label,
        "model_path": str(model_path.resolve()),
        "safetensors_sha256": shas,
        "max_tokens": args.max_tokens,
    }


def cmd_run(args):
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.generate import generate_step

    prompts = read_prompts(args.prompts)
    model, tokenizer = mlx_lm.load(args.model)

    def sampler(lp):
        return mx.argmax(lp, axis=-1)

    with open(args.out, "w") as out:
        out.write(json.dumps(env_record(mx, mlx_lm, args), default=str) + "\n")
        for p in prompts:
            prompt_ids = list(tokenizer.encode(p["text"]))
            tokens, logprobs, margins = [], [], []
            gen = generate_step(mx.array(prompt_ids), model,
                                max_tokens=args.max_tokens, sampler=sampler)
            for tok, lp in gen:
                tok = tok if isinstance(tok, int) else int(tok.item())
                lp = lp.astype(mx.float32).reshape(-1)
                top2 = mx.topk(lp, 2)
                mx.eval(top2, lp)
                a, b = (float(x) for x in top2.tolist())
                hi, lo = max(a, b), min(a, b)
                tokens.append(tok)
                logprobs.append(float(lp[tok].item()))
                margins.append(hi - lo)
            rec = {
                "id": p["id"],
                "prompt_ids": prompt_ids,
                "tokens": tokens,
                "logprobs": logprobs,
                "margins": margins,
                "text": tokenizer.decode(tokens),
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{p['id']}: {len(prompt_ids)} prompt ids, {len(tokens)} tokens", file=sys.stderr)
    return 0


def read_trace(path):
    env, recs = None, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "env":
                env = rec
            else:
                recs[rec["id"]] = rec
    return env, recs


def describe_env(env):
    if not env:
        return "(no env record)"
    return (f"label={env.get('label')!r} device={env.get('default_device')} "
            f"mlx={env.get('mlx_version')} mlx_lm={env.get('mlx_lm_version')} "
            f"platform={env.get('platform')}")


def cmd_compare(args):
    env_a, a = read_trace(args.a)
    env_b, b = read_trace(args.b)
    print(f"A: {args.a}  {describe_env(env_a)}")
    print(f"B: {args.b}  {describe_env(env_b)}")
    if env_a and env_b:
        sa, sb = env_a.get("safetensors_sha256"), env_b.get("safetensors_sha256")
        if sa != sb:
            print("WARNING: safetensors sha256 sets differ between A and B (different weights?)")
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a:
        print(f"note: ids only in A: {only_a}")
    if only_b:
        print(f"note: ids only in B: {only_b}")
    common = [i for i in a if i in b]
    if not common:
        print("error: no common prompt ids")
        return 2

    rows, any_div, any_bad = [], False, False
    for pid in common:
        ra, rb = a[pid], b[pid]
        if ra["prompt_ids"] != rb["prompt_ids"]:
            any_bad = True
            rows.append((pid, "PROMPT_IDS_DIFFER", "-", "-", "-", "-", "-", "-",
                         f"lenA={len(ra['prompt_ids'])} lenB={len(rb['prompt_ids'])}"))
            continue
        ta, tb = ra["tokens"], rb["tokens"]
        n = min(len(ta), len(tb))
        k = next((i for i in range(n) if ta[i] != tb[i]), n)
        if k == n and len(ta) == len(tb):
            rows.append((pid, "MATCH", k, "-", "-", "-", "-", "-", ""))
            continue
        any_div = True
        if k == n:  # one is a prefix of the other
            rows.append((pid, "LENGTH_DIFFERS", k, "-", "-", "-", "-", "-",
                         f"lenA={len(ta)} lenB={len(tb)}"))
            continue
        margins_a, margins_b = ra.get("margins"), rb.get("margins")
        if (
            margins_a is None
            or margins_b is None
            or k >= len(margins_a)
            or k >= len(margins_b)
        ):
            rows.append((pid, "DIVERGE", k, k, ta[k], tb[k], "-", "-", ""))
        else:
            ma, mb = margins_a[k], margins_b[k]
            tie = ma < NEAR_TIE or mb < NEAR_TIE
            rows.append((pid, "DIVERGE", k, k, ta[k], tb[k], f"{ma:.4f}", f"{mb:.4f}",
                         "near-tie" if tie else "NOT a near tie"))

    hdr = ("id", "status", "match_before_div", "div_idx", "A_tok", "B_tok", "A_margin", "B_margin", "note")
    table = [hdr] + [tuple(str(x) for x in r) for r in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(hdr))]
    for r in table:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    n_match = sum(1 for r in rows if r[1] == "MATCH")
    print(f"\nsummary: {n_match}/{len(rows)} prompts match fully; "
          f"near tie threshold = {NEAR_TIE} nats")
    if any_bad:
        print("RESULT: INVALID (prompt_ids differ for at least one prompt)")
        return 2
    if any_div:
        print("RESULT: DIVERGE")
        return 1
    print("RESULT: MATCH")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate a greedy trace")
    r.add_argument("--model", required=True)
    r.add_argument("--prompts", required=True)
    r.add_argument("--max-tokens", type=int, default=64)
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="")
    c = sub.add_parser("compare", help="compare two traces")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    return cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
