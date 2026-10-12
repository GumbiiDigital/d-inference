"""diag.py <cpu|gpu> <out.json>: refgen plus the top-1/top-2 logprob gap at each step, and warm tok/s."""
import json, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
dev = sys.argv[1]
mx.set_default_device(mx.cpu if dev == "cpu" else mx.gpu)
model, tok = load("mlx-community/Qwen3-0.6B-4bit", revision="73e3e38d981303bc594367cd910ea6eb48349da8")
ids = tok.encode("The history of the printing press begins in")
argmax = lambda l: mx.argmax(l, axis=-1)
def run():
    gen, gaps = [], []
    t0 = time.perf_counter()
    for t, lp in generate_step(mx.array(ids), model, max_tokens=128, sampler=argmax):
        gen.append(int(t))
        top = mx.sort(lp.flatten())[-2:]
        gaps.append(round(float(top[1] - top[0]), 4))
    return gen, gaps, time.perf_counter() - t0
g1, gaps, t1 = run()
g2, _, t2 = run()
json.dump({"device": dev, "gen_ids": g1, "gaps": gaps, "run2_same": g1 == g2,
           "tok_s_run1": round(128 / t1, 1), "tok_s_run2": round(128 / t2, 1)}, open(sys.argv[2], "w"))
