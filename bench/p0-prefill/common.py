#!/usr/bin/env python3
"""Shared client library for the P0 batched-prefill validation harness.

Pure stdlib (http.client, threading, json) — zero installs on astra-01.
Targets the ATLAS OpenAI-compatible API (crates/atlas-api):

  POST /v1/chat/completions   SSE stream, `temperature: 0.0` = greedy
  GET  /v1/models             model discovery (no auth)

SSE conventions handled here (see atlas-api/src/types.rs::StreamChunk):
  * `: keep-alive` comment lines during long prefills — ignored for TTFT
  * reasoning models emit `delta.reasoning` before `delta.content` — the
    FIRST token of either kind counts as first token for TTFT
  * usage {prompt_tokens, completion_tokens} arrives on the final content
    chunk; `data: [DONE]` terminates the stream
  * early-429 + Retry-After when the server's max_inflight gate is hit
"""

import http.client
import json
import ssl
import threading
import time
import urllib.parse

# Marker inserted between reasoning text and content text when collecting
# output, so boundary shifts can't silently mask divergences.
REASONING_BOUNDARY = "\x1e"

# Filler words that are single BPE tokens (with leading space) in
# essentially every modern vocab. Cycled deterministically so every prompt
# in a batch has an identical token count.
_FILLER = ["the", "of", "and", "to", "in", "is", "on", "at",
           "by", "for", "with", "from", "as", "or", "an", "be"]


def make_prompt(index, target_tokens):
    """Deterministic prompt #index with ~target_tokens tokens.

    Design constraints (from the batched-prefill eligibility gates):
      * UNIQUE leading tag per prompt -> defeats prefix caching between
        requests (prefix cache disables kernel batching).
      * identical token structure after the tag -> all prompts in a batch
        have EQUAL token counts (co-dispatch / kernel-batch eligibility
        requires equal chunk geometry). The tag is three space-separated
        single digits ("0 4 2") — a leading-space single digit is one BPE
        token in every mainstream vocab, so the tag is always exactly
        3 tokens regardless of index.
    Absolute token count is approximate (vocab-dependent); the true
    prompt_tokens is read back from the server's usage stats.
    """
    tag = " ".join("%03d" % (index % 1000))
    n_filler = max(0, int(target_tokens) - 26)
    filler = " ".join(_FILLER[i % len(_FILLER)] for i in range(n_filler))
    return ("Request {t}: ignore the following filler words: {f}\n"
            "Now answer concisely: list the first five prime numbers."
            ).format(t=tag, f=filler)


class RequestResult(object):
    """Timing + output of one streamed chat completion."""

    def __init__(self):
        self.ok = False
        self.error = None            # str on failure
        self.status = None           # final HTTP status
        self.retries_429 = 0
        self.t_send = None           # monotonic, just before request bytes go out
        self.t_headers = None        # response headers received
        self.t_first = None          # first non-empty token delta
        self.t_last = None           # last token delta
        self.chunks = 0              # token deltas received
        self.prompt_tokens = None    # from usage (server-reported)
        self.completion_tokens = None
        self.text = ""               # reasoning + \x1e + content
        self._seen_content = False

    @property
    def ttft(self):
        if self.t_first is None or self.t_send is None:
            return None
        return self.t_first - self.t_send

    @property
    def decode_tps(self):
        """Steady-state decode rate: tokens after the first, over decode wall."""
        if self.t_last is None or self.t_first is None:
            return None
        n = (self.completion_tokens or self.chunks or 1) - 1
        dt = self.t_last - self.t_first
        if n <= 0 or dt <= 0:
            return None
        return n / dt

    def as_dict(self):
        return {
            "ok": self.ok, "error": self.error, "status": self.status,
            "retries_429": self.retries_429,
            "ttft_s": round(self.ttft, 4) if self.ttft is not None else None,
            "decode_tps": round(self.decode_tps, 2) if self.decode_tps is not None else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "chunks": self.chunks,
            "t_send": self.t_send, "t_first": self.t_first, "t_last": self.t_last,
        }


class AtlasClient(object):
    def __init__(self, base_url, api_key=None, timeout=600.0):
        u = urllib.parse.urlparse(base_url)
        if u.scheme not in ("http", "https"):
            raise ValueError("base_url must be http(s)://host[:port][/prefix]")
        self.scheme = u.scheme
        self.host = u.hostname
        self.port = u.port or (443 if u.scheme == "https" else 80)
        self.prefix = u.path.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ── low-level ────────────────────────────────────────────────────────
    def _conn(self):
        if self.scheme == "https":
            return http.client.HTTPSConnection(
                self.host, self.port, timeout=self.timeout,
                context=ssl.create_default_context())
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def _headers(self, extra=None):
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
        if extra:
            h.update(extra)
        return h

    # ── discovery ────────────────────────────────────────────────────────
    def list_models(self):
        c = self._conn()
        try:
            c.request("GET", self.prefix + "/v1/models", headers=self._headers())
            r = c.getresponse()
            body = r.read().decode("utf-8", "replace")
            if r.status != 200:
                raise RuntimeError("GET /v1/models -> %d: %s" % (r.status, body[:200]))
            data = json.loads(body)
            return [m.get("id") for m in data.get("data", []) if m.get("id")]
        finally:
            c.close()

    # ── streamed chat completion ─────────────────────────────────────────
    def chat_stream(self, prompt, model, max_tokens=64, temperature=0.0,
                    max_429_retries=8, cancel=None):
        """One streamed request. Retries on 429 (honouring Retry-After).

        Returns a RequestResult. Never raises for server-side errors — they
        are captured in .error so batch runs always produce a full table.
        """
        res = RequestResult()
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        })
        attempt = 0
        while True:
            if cancel is not None and cancel.is_set():
                res.error = "cancelled"
                return res
            attempt += 1
            c = self._conn()
            try:
                res.t_send = time.monotonic()
                c.request("POST", self.prefix + "/v1/chat/completions",
                          body=body, headers=self._headers())
                r = c.getresponse()
                res.status = r.status
                res.t_headers = time.monotonic()
                if r.status == 429:
                    retry_after = r.getheader("Retry-After")
                    r.read()
                    res.retries_429 += 1
                    if res.retries_429 > max_429_retries:
                        res.error = "429 retries exhausted (server max_inflight gate; " \
                                    "raise ATLAS_MAX_INFLIGHT >= batch size)"
                        return res
                    try:
                        delay = max(0.5, float(retry_after or 2))
                    except ValueError:
                        delay = 2.0
                    time.sleep(min(delay, 15.0))
                    continue
                if r.status != 200:
                    res.error = "HTTP %d: %s" % (r.status, r.read()[:300].decode("utf-8", "replace"))
                    return res
                self._consume_sse(r, res)
                res.ok = res.error is None
                return res
            except Exception as e:  # noqa: BLE001 — report, don't crash the batch
                res.error = "%s: %s" % (type(e).__name__, e)
                return res
            finally:
                c.close()

    def _consume_sse(self, r, res):
        reasoning_parts, content_parts = [], []
        while True:
            line = r.readline()
            if not line:
                res.error = res.error or "stream ended without [DONE]"
                break
            line = line.decode("utf-8", "replace").strip()
            if not line or line.startswith(":"):
                continue                       # blank / keep-alive comment
            if not line.startswith("data:"):
                continue                       # ignore event:/id: fields
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            choices = obj.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            text = delta.get("content")
            is_reasoning = False
            if text is None:
                text = delta.get("reasoning")
                is_reasoning = text is not None
            usage = obj.get("usage")
            if usage:
                res.prompt_tokens = usage.get("prompt_tokens")
                res.completion_tokens = usage.get("completion_tokens")
            if text:
                now = time.monotonic()
                if res.t_first is None:
                    res.t_first = now
                res.t_last = now
                res.chunks += 1
                if is_reasoning:
                    reasoning_parts.append(text)
                else:
                    content_parts.append(text)
        res.text = "".join(reasoning_parts)
        if content_parts:
            if reasoning_parts:
                res.text += REASONING_BOUNDARY
            res.text += "".join(content_parts)


# ── batch runners ─────────────────────────────────────────────────────────

def run_sequential(client, prompts, model, max_tokens, temperature=0.0):
    """Requests issued strictly one at a time (single-stream prefill path)."""
    t0 = time.monotonic()
    results = []
    for p in prompts:
        results.append(client.chat_stream(p, model, max_tokens, temperature))
    return results, time.monotonic() - t0, t0


def run_concurrent(client, prompts, model, max_tokens, temperature=0.0):
    """All requests fired simultaneously (barrier-aligned) so the server
    sees them together and — with batched prefill wired — co-batches them."""
    n = len(prompts)
    results = [None] * n
    barrier = threading.Barrier(n + 1)

    def worker(i, prompt):
        barrier.wait()
        results[i] = client.chat_stream(prompt, model, max_tokens, temperature)

    threads = [threading.Thread(target=worker, args=(i, p), daemon=True)
               for i, p in enumerate(prompts)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.monotonic()
    for t in threads:
        t.join()
    return results, time.monotonic() - t0, t0


# ── aggregation ───────────────────────────────────────────────────────────

def _pctl(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def aggregate(results, wall_s, t0):
    """Roll one batch's RequestResults into summary stats."""
    ok = [r for r in results if r and r.ok]
    ttfts = sorted(r.ttft for r in ok if r.ttft is not None)
    dtps = [r.decode_tps for r in ok if r.decode_tps is not None]
    ptoks = [r.prompt_tokens for r in ok if r.prompt_tokens]
    # Aggregate prefill throughput: all prompt tokens / time until the LAST
    # request produced its first token (measured from batch start).
    prefill_tps = None
    firsts = [r.t_first for r in ok if r.t_first is not None]
    if ptoks and firsts:
        prefill_wall = max(firsts) - t0
        if prefill_wall > 0:
            prefill_tps = sum(ptoks) / prefill_wall
    return {
        "n": len(results),
        "ok": len(ok),
        "errors": [r.error for r in results if r and r.error],
        "retries_429": sum(r.retries_429 for r in results if r),
        "prompt_tokens": ptoks,
        "prompt_tokens_equal": len(set(ptoks)) <= 1,
        "ttft_p50_s": round(_pctl(ttfts, 0.50), 3) if ttfts else None,
        "ttft_mean_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
        "ttft_max_s": round(max(ttfts), 3) if ttfts else None,
        "prefill_tps_agg": round(prefill_tps, 1) if prefill_tps else None,
        "decode_tps_mean": round(sum(dtps) / len(dtps), 1) if dtps else None,
        "wall_s": round(wall_s, 3),
    }


def resolve_api_key(args):
    """--api-key > --api-key-file > $ATLAS_API_KEY > $ATLAS_API_KEY_FILE."""
    import os
    if getattr(args, "api_key", None):
        return args.api_key
    path = getattr(args, "api_key_file", None) or os.environ.get("ATLAS_API_KEY_FILE")
    if path:
        with open(path) as f:
            return f.read().strip()
    return os.environ.get("ATLAS_API_KEY") or None


def pick_model(client, requested):
    if requested:
        return requested
    models = client.list_models()
    if not models:
        raise RuntimeError("server reported no models; pass --model explicitly")
    return models[0]
