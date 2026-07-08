#!/usr/bin/env python3
"""P0 batched-prefill benchmark: TTFT + throughput, sequential vs batched.

For every batch size B it measures:
  * sequential — B requests one at a time (single-stream prefill baseline)
  * concurrent — B requests fired simultaneously (batched prefill path)

and reports TTFT (p50/max), aggregate prefill tok/s, decode tok/s and wall
time as a markdown table. Results are also written to JSON so a before/after
comparison can be produced with `--compare before.json after.json`.

Examples (on astra-01):
  python3 bench_ttft.py --base-url http://127.0.0.1:8080 \
      --api-key-file ~/.config/atlas/api-key \
      --prompt-tokens 1200 --max-tokens 32 --batch-sizes 1,2,4,8 \
      --label before-wiring -o before.json

  # after the wiring lands (restart server with ATLAS_MAX_INFLIGHT>=8):
  python3 bench_ttft.py ... --label after-wiring -o after.json
  python3 bench_ttft.py --compare before.json after.json

Stdlib only. Exit code: 0 if every request in every cell succeeded.
"""

import argparse
import json
import sys
import time

from common import (AtlasClient, aggregate, make_prompt, pick_model,
                    resolve_api_key, run_concurrent, run_sequential)


def fmt(v, unit=""):
    return ("%.2f%s" % (v, unit)) if isinstance(v, (int, float)) else "—"


def print_table(runs, out=sys.stdout):
    hdr = ("| mode | batch | ok | 429 retries | prompt toks (per req) | "
           "TTFT p50 (s) | TTFT max (s) | prefill tok/s (agg) | "
           "decode tok/s (mean) | wall (s) |")
    sep = "|" + "---|" * 10
    out.write(hdr + "\n" + sep + "\n")
    for r in runs:
        a = r["agg"]
        ptok = a["prompt_tokens"][0] if a["prompt_tokens"] else None
        ptok_s = str(ptok) if ptok else "—"
        if not a.get("prompt_tokens_equal", True):
            ptok_s += " ⚠︎unequal"
        out.write("| %s | %d | %d/%d | %d | %s | %s | %s | %s | %s | %s |\n" % (
            r["mode"], r["batch"], a["ok"], a["n"], a["retries_429"], ptok_s,
            fmt(a["ttft_p50_s"]), fmt(a["ttft_max_s"]),
            fmt(a["prefill_tps_agg"]), fmt(a["decode_tps_mean"]),
            fmt(a["wall_s"])))
    out.flush()


def print_compare(before, after, out=sys.stdout):
    def index(doc):
        return {(r["mode"], r["batch"]): r["agg"] for r in doc["runs"]}
    bi, ai = index(before), index(after)
    out.write("\n### Before/after comparison — %s → %s\n\n" % (
        before["meta"].get("label", "before"), after["meta"].get("label", "after")))
    out.write("| mode | batch | TTFT p50 before | TTFT p50 after | speedup | "
              "prefill tok/s before | prefill tok/s after | gain |\n")
    out.write("|" + "---|" * 8 + "\n")
    for key in sorted(set(bi) | set(ai), key=lambda k: (k[0], k[1])):
        b, a = bi.get(key), ai.get(key)
        bt = b and b.get("ttft_p50_s")
        at_ = a and a.get("ttft_p50_s")
        bp = b and b.get("prefill_tps_agg")
        ap = a and a.get("prefill_tps_agg")
        speed = ("%.1f×" % (bt / at_)) if (bt and at_) else "—"
        gain = ("%.1f×" % (ap / bp)) if (bp and ap) else "—"
        out.write("| %s | %d | %s | %s | %s | %s | %s | %s |\n" % (
            key[0], key[1], fmt(bt), fmt(at_), speed, fmt(bp), fmt(ap), gain))
    out.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--api-key")
    p.add_argument("--api-key-file")
    p.add_argument("--model", help="default: first model from /v1/models")
    p.add_argument("--prompt-tokens", type=int, default=1200,
                   help="target prompt length in tokens (default 1200 — the P0 metric)")
    p.add_argument("--max-tokens", type=int, default=32,
                   help="short decode keeps the benchmark prefill-dominated")
    p.add_argument("--batch-sizes", default="1,2,4,8")
    p.add_argument("--modes", default="sequential,concurrent",
                   help="comma subset of sequential,concurrent")
    p.add_argument("--warmup", type=int, default=1,
                   help="throwaway short requests before measuring (default 1)")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="per-request socket timeout, seconds (100s-class TTFTs are expected pre-wiring)")
    p.add_argument("--cooldown", type=float, default=2.0,
                   help="pause between cells, seconds")
    p.add_argument("--label", default="", help="free-form tag stored in the JSON meta")
    p.add_argument("-o", "--output", help="write results JSON here")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE.json", "AFTER.json"),
                   help="print before/after table from two result files and exit")
    args = p.parse_args()

    if args.compare:
        with open(args.compare[0]) as f:
            before = json.load(f)
        with open(args.compare[1]) as f:
            after = json.load(f)
        print_compare(before, after)
        return 0

    client = AtlasClient(args.base_url, resolve_api_key(args), timeout=args.timeout)
    model = pick_model(client, args.model)
    batches = sorted({int(b) for b in args.batch_sizes.split(",") if b.strip()})
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print("# P0 prefill benchmark — model=%s url=%s prompt≈%d toks max_tokens=%d"
          % (model, args.base_url, args.prompt_tokens, args.max_tokens))

    for i in range(args.warmup):
        print("warmup %d/%d ..." % (i + 1, args.warmup))
        r = client.chat_stream(make_prompt(100 + i, 64), model, max_tokens=8)
        if not r.ok:
            print("  warmup failed: %s" % r.error, file=sys.stderr)

    runs = []
    failed = 0
    prompt_seq = 0
    for batch in batches:
        for mode in modes:
            if batch == 1 and mode == "concurrent" and "sequential" in modes:
                continue  # identical to sequential@1
            # Fresh, unique-prefix prompts per cell -> no prefix-cache hits.
            prompts = [make_prompt(prompt_seq + k, args.prompt_tokens) for k in range(batch)]
            prompt_seq += batch
            print("running %s batch=%d ..." % (mode, batch), flush=True)
            runner = run_sequential if mode == "sequential" else run_concurrent
            results, wall, t0 = runner(client, prompts, model, args.max_tokens)
            agg = aggregate(results, wall, t0)
            failed += agg["n"] - agg["ok"]
            for e in agg["errors"]:
                print("  ! %s" % e, file=sys.stderr)
            runs.append({"mode": mode, "batch": batch, "agg": agg,
                         "requests": [r.as_dict() for r in results]})
            time.sleep(args.cooldown)

    doc = {"meta": {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "base_url": args.base_url, "model": model,
                    "prompt_tokens_target": args.prompt_tokens,
                    "max_tokens": args.max_tokens, "label": args.label},
           "runs": runs}
    print()
    print_table(runs)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(doc, f, indent=1)
        print("\nresults JSON -> %s" % args.output)
    if failed:
        print("\n%d request(s) FAILED — table above is partial" % failed, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
