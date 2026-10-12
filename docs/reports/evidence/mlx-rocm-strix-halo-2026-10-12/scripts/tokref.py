"""tokref.py MODEL REVISION DEVICE OUT.json [N]: greedy N tokens (default 128) with mlx-lm on a chat-template prompt, plus the top-1/top-2
logprob gap at each step, for a cross-device token match with a tie tolerance. DEVICE is gpu or cpu."""
import json, platform, sys, time
import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import generate_step

model_id, revision, dev, out = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 128
mx.set_default_device(mx.cpu if dev == "cpu" else mx.gpu)
model, tok = load(model_id, revision=revision)
prompt = "Tell me about the history of the printing press."
# Chat template: instruction-tuned models (Gemma 4 IT) degenerate on a raw prompt without it.
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
gen, top2, gaps = [], [], []
t0 = time.perf_counter()
for t, lp in generate_step(mx.array(ids), model, max_tokens=n, sampler=lambda l: mx.argmax(l, axis=-1)):
    lp = lp.flatten()
    i = mx.argsort(lp)[-2:]
    gen.append(int(t)); top2.append([int(i[1]), int(i[0])]); gaps.append(round(float(lp[i[1]] - lp[i[0]]), 4))
dt = time.perf_counter() - t0
json.dump({"host": platform.platform(), "device": str(mx.default_device()), "mlx": mx.__version__,
           "mlx_lm": mlx_lm.__version__, "model": model_id, "revision": revision, "prompt": prompt,
           "prompt_ids": ids, "gen_ids": gen, "top2": top2, "gaps": gaps, "gen_text": tok.decode(gen),
           "tok_per_s_incl_prefill": round(n / dt, 2)}, open(out, "w"), indent=1)
print(tok.decode(gen))
