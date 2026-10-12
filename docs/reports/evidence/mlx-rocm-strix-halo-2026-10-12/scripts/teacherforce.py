"""teacherforce.py REF.json DEVICE OUT.json: teacher-forced token check against a reference run from tokref.py.
Feeds the reference prompt, then each reference token one at a time through the decode path (KV cache), and at each
step compares this device's argmax with the reference token. A mismatch counts as a tie only if this device's
logprob of the reference token equals its top logprob (exact tie). Reports matches, ties and real mismatches."""
import json, platform, sys
import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

ref = json.load(open(sys.argv[1]))
mx.set_default_device(mx.cpu if sys.argv[2] == "cpu" else mx.gpu)
model, tok = load(ref["model"], revision=ref["revision"])
cache = make_prompt_cache(model)

def step(ids):
    logits = model(mx.array(ids)[None], cache=cache)[0, -1].astype(mx.float32)
    lp = logits - mx.logsumexp(logits)
    mx.eval(lp)
    return lp

lp = step(ref["prompt_ids"])
rows, match, ties, bad = [], 0, 0, []
for i, t in enumerate(ref["gen_ids"]):
    top = int(mx.argmax(lp))
    gap = float(lp[top] - lp[t])  # 0 when the reference token is (one of) this device's top tokens
    if top == t:
        match += 1
    elif gap == 0.0:
        ties += 1
    else:
        bad.append({"step": i, "ref": t, "here": top, "gap": round(gap, 4)})
    rows.append({"step": i, "ref": t, "here": top, "gap": round(gap, 4)})
    lp = step([t])
out = {"host": platform.platform(), "device": str(mx.default_device()), "mlx": mx.__version__,
       "mlx_lm": mlx_lm.__version__, "model": ref["model"], "revision": ref["revision"], "ref_device": ref["device"],
       "ref_mlx": ref["mlx"], "steps": len(ref["gen_ids"]), "argmax_match": match, "exact_ties": ties,
       "mismatches": bad, "rows": rows}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(f"steps {out['steps']} match {match} exact_ties {ties} mismatches {len(bad)} {bad[:5]}")
