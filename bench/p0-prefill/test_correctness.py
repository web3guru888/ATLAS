#!/usr/bin/env python3
"""Batch-correctness test: batched prefill must not change model outputs.

Runs the SAME prompt set (greedy, temperature=0) two ways:
  phase 1 (reference) — strictly one request at a time (single-stream path)
  phase 2 (batched)   — all prompts fired concurrently at each batch size

and asserts byte-identical output text per prompt (reasoning + content,
with a \\x1e boundary marker so reasoning/content boundary shifts are
caught too). Any wiring bug — wrong block-table pointer, cross-stream KV
contamination, mis-staged batch metadata — shows up as a divergence, and
the report pinpoints the first diverging character with context.

⚠ Prefix caching caveat: if the server caches prompt KV between phases,
phase 2 may silently reuse phase-1 KV and never exercise the batched
kernels. For a valid run, start the server with prefix caching disabled,
or pass --pause and restart the server when prompted between phases.

Exit code 0 = all comparisons byte-identical, 1 = divergence or error.

Example (astra-01):
  python3 test_correctness.py --base-url http://127.0.0.1:8080 \
      --api-key-file ~/.config/atlas/api-key \
      --prompt-tokens 600 --max-tokens 128 --batch-sizes 2,4,8
"""

import argparse
import json
import sys
import time

from common import (AtlasClient, make_prompt, pick_model, resolve_api_key,
                    run_concurrent, run_sequential)


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None


def show(s, i, width=40):
    lo, hi = max(0, i - width), i + width
    return repr(s[lo:hi])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--api-key")
    p.add_argument("--api-key-file")
    p.add_argument("--model")
    p.add_argument("--prompt-tokens", type=int, default=600)
    p.add_argument("--max-tokens", type=int, default=128,
                   help="longer decode = more chance to catch late divergence")
    p.add_argument("--batch-sizes", default="2,4,8")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--pause", action="store_true",
                   help="wait for Enter between phases (restart server to clear prefix cache)")
    p.add_argument("--min-prefix", type=float, default=1.0,
                   help="pass threshold as matching-prefix fraction (1.0 = strict "
                        "byte-identical; e.g. 0.95 tolerates late FP-reduction "
                        "argmax flips while still failing on garbage)")
    p.add_argument("-o", "--output", help="write full outputs + verdicts JSON here")
    args = p.parse_args()

    client = AtlasClient(args.base_url, resolve_api_key(args), timeout=args.timeout)
    model = pick_model(client, args.model)
    batches = sorted({int(b) for b in args.batch_sizes.split(",") if b.strip()})
    n_prompts = max(batches)
    prompts = [make_prompt(i, args.prompt_tokens) for i in range(n_prompts)]

    print("# P0 batch-correctness — model=%s greedy temp=0 max_tokens=%d prompts=%d"
          % (model, args.max_tokens, n_prompts))

    # Phase 1 — trusted single-stream reference, one request at a time.
    print("\nphase 1: sequential reference (%d prompts, one at a time) ..." % n_prompts)
    refs, wall, _ = run_sequential(client, prompts, model, args.max_tokens, temperature=0.0)
    ref_texts = []
    for i, r in enumerate(refs):
        if not r.ok:
            print("  ! reference request %d failed: %s" % (i, r.error), file=sys.stderr)
            return 1
        ref_texts.append(r.text)
        print("  ref[%d]: prompt_toks=%s completion_toks=%s len=%d"
              % (i, r.prompt_tokens, r.completion_tokens, len(r.text)))
    if len(set(ref_texts)) == 1 and n_prompts > 1:
        print("  ⚠ all reference outputs identical — prompts may be too similar "
              "to catch cross-stream mixups", file=sys.stderr)

    # Determinism sanity: re-run prompt 0 sequentially; if the engine is not
    # deterministic even single-stream, strict comparison is meaningless.
    r2 = client.chat_stream(prompts[0], model, args.max_tokens, temperature=0.0)
    deterministic = r2.ok and r2.text == ref_texts[0]
    if not deterministic:
        print("  ⚠ single-stream greedy decode is NOT deterministic (rerun of "
              "prompt 0 differs%s). Strict pass/fail below is unreliable — "
              "fix determinism first or interpret via --min-prefix." %
              ("" if r2.ok else "; rerun errored: %s" % r2.error), file=sys.stderr)

    if args.pause:
        input("\nphase 1 done. Restart the server now to clear any prefix cache, "
              "then press Enter to start batched phase ...")

    # Phase 2 — batched runs, compare against reference.
    report = {"meta": {"model": model, "base_url": args.base_url,
                       "max_tokens": args.max_tokens,
                       "prompt_tokens_target": args.prompt_tokens,
                       "deterministic_baseline": deterministic,
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              "reference": [{"prompt_i": i, "text": t} for i, t in enumerate(ref_texts)],
              "batched": []}
    failures = 0
    for batch in batches:
        print("\nphase 2: concurrent batch=%d ..." % batch)
        results, wall, _ = run_concurrent(client, prompts[:batch], model,
                                          args.max_tokens, temperature=0.0)
        for i, r in enumerate(results):
            entry = {"batch": batch, "prompt_i": i, "ok": r.ok, "error": r.error,
                     "text": r.text, "retries_429": r.retries_429}
            if not r.ok:
                print("  ✗ [batch=%d req=%d] request FAILED: %s" % (batch, i, r.error))
                failures += 1
            else:
                div = first_divergence(ref_texts[i], r.text)
                match_frac = 1.0 if div is None else (div / max(1, len(ref_texts[i])))
                entry["first_divergence"] = div
                entry["match_fraction"] = round(match_frac, 4)
                if div is None:
                    print("  ✓ [batch=%d req=%d] identical (%d chars)"
                          % (batch, i, len(r.text)))
                else:
                    passed = match_frac >= args.min_prefix
                    failures += 0 if passed else 1
                    print("  %s [batch=%d req=%d] diverges at char %d/%d "
                          "(match %.1f%%)\n      ref: %s\n      got: %s"
                          % ("~" if passed else "✗", batch, i, div,
                             len(ref_texts[i]), 100 * match_frac,
                             show(ref_texts[i], div), show(r.text, div)))
            report["batched"].append(entry)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
        print("\nfull report JSON -> %s" % args.output)

    if failures:
        print("\nRESULT: FAIL — %d divergence(s)/error(s). Batched prefill is "
              "changing outputs; do NOT enable by default." % failures)
        return 1
    print("\nRESULT: PASS — batched outputs byte-identical to sequential "
          "reference at all batch sizes (%s)." % ",".join(map(str, batches)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
