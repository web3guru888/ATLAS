# P0 Batched-Prefill Validation Harness

Proves (or disproves) that the `feat/batched-prefill-wire` wiring works:
**(1)** TTFT/throughput actually improves under concurrent load, and
**(2)** batched prefill produces **byte-identical greedy outputs** vs the
trusted single-stream path.

Pure Python 3.8+ stdlib — **zero installs** on astra-01. Targets the
OpenAI-compatible ATLAS API (`/v1/chat/completions`, SSE).

| File | Purpose |
|---|---|
| `common.py` | stdlib SSE client, equal-token-length prompt generator, batch runners |
| `bench_ttft.py` | TTFT + throughput: sequential vs concurrent at batch 1/2/4/8, before/after compare |
| `test_correctness.py` | greedy (temp=0) parity: batched vs sequential must match byte-for-byte |
| `mock_server.py` | GPU-free mock of the API for harness self-tests / CI |

## Quick start on astra-01

```bash
cd ATLAS/bench/p0-prefill

# BEFORE the wiring (baseline — expect ~100s-class TTFT at 1.2K on 32B):
python3 bench_ttft.py --base-url http://127.0.0.1:8080 \
    --api-key-file ~/.config/atlas/api-key \
    --prompt-tokens 1200 --max-tokens 32 --batch-sizes 1,2,4,8 \
    --label before-wiring -o /tmp/before.json

# ... land the wiring, restart the server with ATLAS_MAX_INFLIGHT >= 8 ...

# AFTER:
python3 bench_ttft.py --base-url http://127.0.0.1:8080 \
    --api-key-file ~/.config/atlas/api-key \
    --prompt-tokens 1200 --max-tokens 32 --batch-sizes 1,2,4,8 \
    --label after-wiring -o /tmp/after.json

# The deliverable table:
python3 bench_ttft.py --compare /tmp/before.json /tmp/after.json

# Correctness gate (MUST pass before enabling by default):
python3 test_correctness.py --base-url http://127.0.0.1:8080 \
    --api-key-file ~/.config/atlas/api-key \
    --prompt-tokens 600 --max-tokens 128 --batch-sizes 2,4,8
```

Notes:
* If the key file needs robindey perms: `sudo -u robindey cat /home/robindey/.config/atlas/api-key`
  and pass via `--api-key` / `ATLAS_API_KEY` env. Never commit keys.
* The server's early-429 gate caps concurrency — for batched runs start it
  with `ATLAS_MAX_INFLIGHT` ≥ the largest batch size, or the harness will
  report 429 retries in the table (it retries with backoff and counts them).
* Pre-wiring TTFTs are long — the default `--timeout 900` allows for it;
  keep-alive SSE comments during prefill are handled (and excluded from TTFT).

## What the benchmark measures

For each batch size B × mode (`sequential` = B requests one at a time;
`concurrent` = B fired simultaneously, barrier-aligned):

* **TTFT** — request sent → first non-empty token delta (`reasoning` or
  `content`; keep-alives ignored), p50/max across the batch
* **prefill tok/s (agg)** — Σ server-reported `prompt_tokens` ÷ (batch start
  → last request's first token). This is the headline batching win.
* **decode tok/s (mean)** — per-request steady-state decode rate
* **429 retries** — nonzero means the inflight gate, not the engine, was the limiter
* **prompt toks** — server-reported; flagged `⚠︎unequal` if the batch wasn't
  equal-length (equal chunk geometry is an eligibility gate for kernel batching)

Prompts are generated with a **unique leading tag** (defeats prefix-cache
hits, which disable kernel batching) and **identical token structure**
(single-token tag variance → equal token counts across the batch).

Expected output shape:

```
| mode | batch | ok | 429 retries | prompt toks (per req) | TTFT p50 (s) | TTFT max (s) | prefill tok/s (agg) | decode tok/s (mean) | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| sequential | 1 | 1/1 | 0 | 1187 | 98.20 | 98.20 | 12.09 | 49.80 | 99.10 |
| concurrent | 4 | 4/4 | 0 | 1187 | 101.30 | 103.10 | 46.05 | 48.90 | 104.40 |
```

and `--compare` produces:

```
| mode | batch | TTFT p50 before | TTFT p50 after | speedup | prefill tok/s before | prefill tok/s after | gain |
```

Success criteria (from the P0 plan): TTFT@1.2K 100s → 2–5s (32B W4),
concurrent prefill tok/s scaling ≈ linearly with batch up to the arena cap.

## What the correctness test guarantees

Phase 1 runs every prompt **strictly sequentially** (trusted single-stream
path) and records full output text (reasoning + `\x1e` + content — the
boundary marker catches reasoning/content boundary shifts). It also re-runs
prompt 0 to verify the engine is deterministic at temp=0 (if not, strict
comparison is flagged as unreliable). Phase 2 fires the same prompts
concurrently at each batch size and diffs each output against its reference,
reporting the first diverging character with ±40 chars of context.

Wiring bugs this catches: cross-stream KV contamination (stream i gets
stream j's continuation), mis-staged block-table pointers (garbage after
the first chunk boundary), wrong batch metadata (divergence at token 1).

* **Strict by default** (`--min-prefix 1.0`). Batched GEMM reductions *can*
  legitimately flip a late argmax vs GEMV order; if you decide to tolerate
  that, `--min-prefix 0.95` still fails hard on real wiring bugs (which
  diverge early), and every divergence index is printed either way.
* **Prefix-cache caveat**: phase 2 must not silently reuse phase-1 KV.
  Run the server with prefix caching disabled, or use `--pause` and restart
  the server between phases.

## Harness self-test (no GPU)

```bash
python3 mock_server.py &                    # :18080, single-stream behaviour
python3 bench_ttft.py  --base-url http://127.0.0.1:18080 \
    --prompt-tokens 400 --max-tokens 16 --batch-sizes 1,2,4 --warmup 0 --cooldown 0
python3 test_correctness.py --base-url http://127.0.0.1:18080 \
    --prompt-tokens 200 --max-tokens 24 --batch-sizes 2,4

# negative control — corrupt concurrent (batched-phase) streams only;
# correctness test MUST fail:
MOCK_FAULT_CONCURRENT=1 MOCK_PORT=18081 python3 mock_server.py &
python3 test_correctness.py --base-url http://127.0.0.1:18081 --batch-sizes 4 \
    --prompt-tokens 200 --max-tokens 24 ; echo "exit=$? (expect 1)"

# MOCK_SERIALIZE=0 emulates post-wiring parallel prefill (TTFT stays flat
# as batch grows instead of stacking) — useful to sanity-check --compare.
```

---
*beast-engineer · 2026-07-08 · ATLAS P0 sprint session 1 · plan:
`/shared/beast-atlas/perf-improvement-plan-20260707.md`*
