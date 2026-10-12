#!/usr/bin/env python3
"""Time MLX (in-process) and llama.cpp (llama-server) the same way, interleaved.

Usage:
  python3 bench_compare.py --mlx-model PATH --gguf PATH --llama-server /path/llama-server \
      --prompt-file prompt.txt --out result.json \
      [--prefill-lens 512,2048] [--decode-tokens 128] [--reps 5] [--warmup 1] \
      [--engines mlx,llamacpp] [--port 18080] [--llama-extra-args "-fa on"] \
      [--gpu-temp-max 60] [--gpu-util-max 10]
  llama.cpp only, without mlx installed: --engines llamacpp --prompt-ids ids.json
      (ids.json: a JSON list of token ids, or an object {"512": [...], "2048": [...]})

Timing (client side, perf_counter_ns, same for both engines):
  prefill tok/s = P / (t_first - t0); decode tok/s = (N - 1) / (t_last - t_first).
  MLX: generate_step schedules token n+1 before it yields token n (real mlx-lm behaviour);
  stamps are taken at each yield.
"""
import argparse, http.client, importlib.util, json, platform, shlex, signal, socket, statistics
import subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path


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


def finalize_gdn_patch_env(env, gdn_mod):
    env["mlx_patch_counts"] = gdn_mod.counts()
    print(f"mlx_patch_counts: {json.dumps(env['mlx_patch_counts'])}", file=sys.stderr)


def now():
    return datetime.now(timezone.utc).isoformat()


def run_text(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERROR: {e!r}"


def gpu_gate(temp_max, util_max):
    """Wait (2 s steps, max 120 s) until temp <= max and util <= max. Return the record."""
    t_start = time.monotonic()
    while True:
        try:
            proc = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu",
                                   "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10)
        except FileNotFoundError:
            return {"temp": None, "util": None, "note": "nvidia-smi missing", "gate_passed": None}
        out = proc.stdout.strip()
        parts = [x.strip() for x in out.splitlines()[0].split(",")] if out else ["", ""]
        vals = []
        for x in parts[:2]:
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(None)
        temp, util = (vals + [None, None])[:2]
        ok = (proc.returncode == 0 and temp is not None and util is not None
              and temp <= temp_max and util <= util_max)
        waited = round(time.monotonic() - t_start, 1)
        if ok or waited >= 120:
            return {"temp": temp, "util": util, "raw": out, "waited_s": waited,
                    "nvidia_smi_rc": proc.returncode, "gate_passed": ok if ok else False}
        time.sleep(2)


def rates(P, N, t0, t_first, t_last):
    pre = P / ((t_first - t0) / 1e9) if t_first > t0 else None
    dec = (N - 1) / ((t_last - t_first) / 1e9) if N > 1 and t_last > t_first else None
    return pre, dec


class MLXEngine:
    def __init__(self, model_path):
        import mlx.core as mx
        import mlx_lm
        from mlx_lm.generate import generate_step
        self.mx, self.mlx_lm, self.generate_step = mx, mlx_lm, generate_step
        self.model, self.tokenizer = mlx_lm.load(model_path)

    def env(self):
        mx = self.mx
        info = None
        if hasattr(mx, "device_info"):
            try:
                info = mx.device_info()
            except Exception as e:
                info = {"error": repr(e)}
        return {"mlx": mx.__version__, "mlx_lm": self.mlx_lm.__version__,
                "default_device": str(mx.default_device()), "device_info": info}

    def rep(self, ids, N):
        mx = self.mx
        sampler = lambda lp: mx.argmax(lp, axis=-1)
        prompt = mx.array(ids)
        stamps, toks = [], []
        t0 = time.perf_counter_ns()
        for tok, _lp in self.generate_step(prompt, self.model, max_tokens=N, sampler=sampler):
            tok = tok if isinstance(tok, int) else int(tok.item())
            stamps.append(time.perf_counter_ns())
            toks.append(tok)
        rec = {"n_tokens": len(toks), "tokens": toks, "token_count_source": "generate_step yields"}
        if len(toks) != N:
            rec.update(valid=False, error=f"got {len(toks)} tokens, expected {N}")
            return rec
        pre, dec = rates(len(ids), N, t0, stamps[0], stamps[-1])
        rec.update(valid=True, t0=t0, t_first=stamps[0], t_last=stamps[-1],
                   prefill_tps=pre, decode_tps=dec)
        return rec


class LlamaEngine:
    def __init__(self, binary, gguf, port, extra, log_path):
        self.port = port
        cmd = [binary, "-m", gguf, "-ngl", "99", "-c", "8192", "-np", "1", "--no-webui",
               "--port", str(port)] + shlex.split(extra)
        self.cmd = cmd
        self.version = run_text([binary, "--version"])
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        deadline = time.monotonic() + 900
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {log_path}")
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"llama-server /health not 200 after 900 s; see {log_path}")
            time.sleep(1)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()

    def rep(self, ids, N):
        body = json.dumps({"prompt": ids, "n_predict": N, "temperature": 0, "top_k": 1,
                           "cache_prompt": False, "ignore_eos": True, "stream": True,
                           "return_tokens": True, "n_probs": 0})
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        t0 = time.perf_counter_ns()
        c.request("POST", "/completion", body=body, headers={"Content-Type": "application/json"})
        resp = c.getresponse()
        if resp.status != 200:
            return {"valid": False, "error": f"HTTP {resp.status}: {resp.read()[:500]!r}"}
        stamps, toks, text, timings, final = [], [], [], None, None
        stream_invalid = None
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            ev = json.loads(line[5:].strip())
            batch = ev["tokens"] if "tokens" in ev else None
            if batch is not None:
                for t in batch:
                    toks.append(t)
            elif ev.get("content"):
                stream_invalid = "no token ids in stream"
            ts = time.perf_counter_ns()
            if batch is not None:
                stamps.extend([ts] * len(batch))
            if ev.get("stop"):
                final = ev
                timings = ev.get("timings")
                text.append(ev.get("content", ""))
                break
            text.append(ev.get("content", ""))
        c.close()
        n = len(toks)
        rec = {"n_tokens": n, "tokens": toks, "text": "".join(text),
               "token_count_source": "tokens list", "engine_timings": timings}
        pred_n = timings.get("predicted_n") if isinstance(timings, dict) else None
        if stream_invalid:
            rec.update(valid=False, reason=stream_invalid, error=stream_invalid)
        elif final is None:
            rec.update(valid=False, error="stream ended without a stop event")
        elif n != N:
            rec.update(valid=False, error=f"streamed {n} tokens, expected {N}")
        elif pred_n is not None and pred_n != N:
            rec.update(valid=False, error=f"timings.predicted_n={pred_n}, expected {N}")
        elif not stamps:
            rec.update(valid=False, error="no token timestamps collected")
        else:
            pre, dec = rates(len(ids), N, t0, stamps[0], stamps[-1])
            rec.update(valid=True, t0=t0, t_first=stamps[0], t_last=stamps[-1],
                       prefill_tps=pre, decode_tps=dec)
        return rec


def build_prompt_ids(args, tokenizer, lens):
    maxp = max(lens)
    if args.prompt_ids:
        data = json.loads(Path(args.prompt_ids).read_text())
        out = {}
        for P in lens:
            src = data.get(str(P)) if isinstance(data, dict) else data
            if src is None or len(src) < P:
                sys.exit(f"error: --prompt-ids has fewer than {P} ids for P={P}")
            out[P] = [int(x) for x in src[:P]]
        return out, "prompt-ids file"
    text = Path(args.prompt_file).read_text()
    ids = list(tokenizer.encode(text))
    k = 1
    while len(ids) < maxp:
        k += 1
        ids = list(tokenizer.encode("\n".join([text] * k)))
        if k > 10000:
            sys.exit("error: prompt text tokenizes to nothing useful")
    return {P: ids[:P] for P in lens}, f"mlx tokenizer, text repeated x{k}"


def summarize(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = statistics.mean(vals)
    cv = statistics.stdev(vals) / m * 100 if len(vals) > 1 and m else None
    return {"n": len(vals), "median": statistics.median(vals), "min": min(vals),
            "max": max(vals), "cv_pct": cv}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlx-model"); ap.add_argument("--gguf"); ap.add_argument("--llama-server")
    ap.add_argument("--prompt-file"); ap.add_argument("--prompt-ids")
    ap.add_argument("--prefill-lens", default="512,2048")
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--engines", default="mlx,llamacpp"); ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--out", required=True); ap.add_argument("--llama-extra-args", default="-fa on")
    ap.add_argument("--gpu-temp-max", type=float, default=60)
    ap.add_argument("--gpu-util-max", type=float, default=10)
    ap.add_argument("--mlx-patch", default=None, help="optional mlx-lm patch (e.g. gdn_cuda)")
    args = ap.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    bad = set(engines) - {"mlx", "llamacpp"}
    if bad or not engines:
        sys.exit(f"error: unknown engines {sorted(bad)}; use mlx and/or llamacpp")
    lens = [int(x) for x in args.prefill_lens.split(",")]
    N = args.decode_tokens
    if "mlx" in engines and not args.mlx_model:
        sys.exit("error: --mlx-model is required for the mlx engine")
    if "llamacpp" in engines and not (args.gguf and args.llama_server):
        sys.exit("error: --gguf and --llama-server are required for the llamacpp engine")
    if not args.prompt_ids and not args.prompt_file:
        sys.exit("error: give --prompt-file (or --prompt-ids for llamacpp-only runs)")
    if not args.prompt_ids and not args.mlx_model:
        sys.exit("error: --prompt-file needs --mlx-model for its tokenizer; or use --prompt-ids")

    def on_term(signum, frame):
        raise SystemExit(f"terminated by signal {signum}")
    signal.signal(signal.SIGTERM, on_term)

    result = {"env": {"start": now(), "hostname": socket.gethostname(), "platform": platform.platform(),
                      "python": sys.version, "args": vars(args)}, "reps": []}
    gdn_mod = install_mlx_patch(args.mlx_patch) if args.mlx_patch else None
    if gdn_mod is not None:
        result["env"]["mlx_patch"] = gdn_mod._INSTALL_REPORT or {"mlx_lm.models.gated_delta": True}
    eng = {}
    try:
        tokenizer = None
        if "mlx" in engines:
            eng["mlx"] = MLXEngine(args.mlx_model)
            if gdn_mod is not None:
                record_gdn_patch_env(result["env"], gdn_mod)
            tokenizer = eng["mlx"].tokenizer
            result["env"]["mlx"] = eng["mlx"].env()
        elif not args.prompt_ids:
            from mlx_lm.utils import load_tokenizer  # tokenizer only, no weights
            tokenizer = load_tokenizer(Path(args.mlx_model))
        prompt_ids, src = build_prompt_ids(args, tokenizer, lens)
        result["env"]["prompt_ids_source"] = src
        result["prompt_ids"] = {str(P): v for P, v in prompt_ids.items()}
        if "llamacpp" in engines:
            log_path = str(Path(args.out).with_suffix("")) + ".llama-server.log"
            eng["llamacpp"] = LlamaEngine(args.llama_server, args.gguf, args.port,
                                          args.llama_extra_args, log_path)
            result["env"]["llama_server"] = {"cmd": eng["llamacpp"].cmd, "version": eng["llamacpp"].version,
                                             "log": log_path}
        result["env"]["nvidia_smi_clocks"] = run_text(["nvidia-smi", "-q", "-d", "CLOCK"])

        def do_rep(e, P, rep, phase):
            gate = gpu_gate(args.gpu_temp_max, args.gpu_util_max) if phase == "timed" else None
            if phase == "timed" and gate is not None and gate.get("gate_passed") is False:
                r = {"valid": False, "reason": "gate", "error": "gpu gate not passed"}
            else:
                try:
                    r = eng[e].rep(prompt_ids[P], N)
                except Exception as ex:
                    r = {"valid": False, "error": f"{type(ex).__name__}: {ex}"}
            r.update(engine=e, P=P, N=N, rep=rep, phase=phase, gpu=gate, time=now())
            pr = f"{r['prefill_tps']:.1f}" if r.get("prefill_tps") else "-"
            dr = f"{r['decode_tps']:.1f}" if r.get("decode_tps") else "-"
            fail = "" if r.get("valid") else f"  FAILED: {r.get('reason') or r.get('error', '')}"
            print(f"[{phase} {rep}] {e:9s} P={P:<6d} prefill={pr:>9s} decode={dr:>8s}{fail}",
                  file=sys.stderr)
            return r

        for e in engines:
            for P in lens:
                for w in range(args.warmup):
                    wr = do_rep(e, P, w + 1, "warmup")
                    if not wr.get("valid"):
                        sys.exit(f"error: warmup failed for {e} P={P} rep={w + 1}: "
                                 f"{wr.get('error', wr.get('reason', 'unknown'))}")
        for P in lens:
            for e in engines:
                try:
                    eng[e].rep(prompt_ids[P], N)
                except Exception as ex:
                    sys.exit(f"error: pre-timed priming rep failed for {e} P={P}: {ex}")
        for rep in range(1, args.reps + 1):
            for P in lens:
                order = engines if rep % 2 == 1 else list(reversed(engines))
                for e in order:
                    result["reps"].append(do_rep(e, P, rep, "timed"))
    finally:
        if "llamacpp" in eng:
            eng["llamacpp"].stop()
        result["env"]["end"] = now()
        if gdn_mod is not None:
            finalize_gdn_patch_env(result["env"], gdn_mod)

    stats, table = {}, []
    for e in engines:
        for P in lens:
            rs = [r for r in result["reps"] if r["engine"] == e and r["P"] == P and r["valid"]]
            nf = sum(1 for r in result["reps"] if r["engine"] == e and r["P"] == P and not r["valid"])
            stats[f"{e}/P{P}"] = {
                "engine": e, "P": P, "failed_reps": nf,
                "prefill_client_side_throughput_tps": summarize([r["prefill_tps"] for r in rs]),
                "decode_client_side_throughput_tps": summarize([r["decode_tps"] for r in rs]),
            }
    ratios, greedy = {}, {}
    for P in lens:
        a, b = stats.get(f"mlx/P{P}"), stats.get(f"llamacpp/P{P}")
        if a and b:
            ratios[f"P{P}"] = {
                "prefill_client_side_throughput": (
                    a["prefill_client_side_throughput_tps"]["median"] / b["prefill_client_side_throughput_tps"]["median"]
                    if a["prefill_client_side_throughput_tps"] and b["prefill_client_side_throughput_tps"]
                    and b["prefill_client_side_throughput_tps"]["median"] else None),
                "decode_client_side_throughput": (
                    a["decode_client_side_throughput_tps"]["median"] / b["decode_client_side_throughput_tps"]["median"]
                    if a["decode_client_side_throughput_tps"] and b["decode_client_side_throughput_tps"]
                    and b["decode_client_side_throughput_tps"]["median"] else None),
            }
            ra = next((r for r in result["reps"] if r["engine"] == "mlx" and r["P"] == P and r["rep"] == 1), None)
            rb = next((r for r in result["reps"] if r["engine"] == "llamacpp" and r["P"] == P and r["rep"] == 1), None)
            if (ra and rb and ra.get("valid") and rb.get("valid")
                    and len(ra.get("tokens") or []) == N and len(rb.get("tokens") or []) == N):
                ta, tb = ra["tokens"], rb["tokens"]
                k = next((i for i in range(N) if ta[i] != tb[i]), N)
                g = {"same_tokens": ta == tb, "match_prefix": k, "compared": "token ids"}
            else:
                parts = []
                if not ra:
                    parts.append("mlx rep 1 missing")
                elif not ra.get("valid"):
                    parts.append(f"mlx rep 1 invalid: {ra.get('reason') or ra.get('error', '?')}")
                elif len(ra.get("tokens") or []) != N:
                    parts.append(f"mlx rep 1 has {len(ra.get('tokens') or [])} tokens, expected {N}")
                if not rb:
                    parts.append("llamacpp rep 1 missing")
                elif not rb.get("valid"):
                    parts.append(f"llamacpp rep 1 invalid: {rb.get('reason') or rb.get('error', '?')}")
                elif len(rb.get("tokens") or []) != N:
                    parts.append(f"llamacpp rep 1 has {len(rb.get('tokens') or [])} tokens, expected {N}")
                g = {"same_tokens": None, "reason": "; ".join(parts) or "rep 1 not comparable"}
            greedy[f"P{P}"] = g
    gate_null_reps = [r for r in result["reps"]
                      if r.get("gpu") and r["gpu"].get("gate_passed") is None]
    result.update(stats=stats, ratio_mlx_over_llamacpp=ratios, greedy_match_rep1=greedy,
                  throughput_label="client-side throughput")
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))

    print("\nclient-side throughput (tok/s, perf_counter_ns at each yielded token)")
    print(f"{'engine':10s} {'P':>6s} {'prefill med':>12s} {'cv%':>6s} {'decode med':>11s} {'cv%':>6s} fail")
    for s in stats.values():
        f = lambda d, k: (f"{d[k]:.1f}" if d and d[k] is not None else "-")
        pre = s["prefill_client_side_throughput_tps"]
        dec = s["decode_client_side_throughput_tps"]
        print(f"{s['engine']:10s} {s['P']:>6d} {f(pre, 'median'):>12s} "
              f"{f(pre, 'cv_pct'):>6s} {f(dec, 'median'):>11s} "
              f"{f(dec, 'cv_pct'):>6s} {s['failed_reps']}")
    if gate_null_reps:
        print(f"warning: {len(gate_null_reps)} timed rep(s) had no gpu gate (nvidia-smi missing); "
              "throughput still recorded")
    for P, r in ratios.items():
        fmt = lambda x: f"{x:.3f}" if x is not None else "-"
        print(f"mlx/llamacpp {P}: prefill {fmt(r['prefill_client_side_throughput'])}  "
              f"decode {fmt(r['decode_client_side_throughput'])}  "
              f"greedy rep1: {greedy.get(P)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
