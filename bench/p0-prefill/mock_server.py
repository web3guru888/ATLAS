#!/usr/bin/env python3
"""Tiny stdlib mock of the ATLAS /v1 API — for harness self-tests only.

Emulates just enough of atlas-api for bench_ttft.py / test_correctness.py
to run without a GPU: SSE streaming, keep-alive comments during "prefill",
reasoning-then-content deltas, usage on the final chunk, [DONE].

Output is DETERMINISTIC per prompt (seeded by prompt hash), so the
correctness test passes against it, and "prefill" time is proportional to
prompt length, so the benchmark produces meaningful-shaped numbers.

Env knobs:
  MOCK_PORT           listen port                     (default 18080)
  MOCK_S_PER_TOK      prefill seconds per prompt tok  (default 0.002)
  MOCK_DECODE_TPS     decode tokens/second            (default 200)
  MOCK_SERIALIZE      1 = single-stream engine (global lock; concurrent
                          requests queue — emulates ATLAS TODAY)
                      0 = parallel prefill (emulates batched wiring)
  MOCK_API_KEY        if set, require this bearer key
  MOCK_FAULT_CONCURRENT  if "1": any request that is in flight TOGETHER
                      with another request gets one corrupted mid-stream
                      token (sequential/reference requests stay clean) —
                      for verifying that test_correctness.py FAILS loudly.
"""

import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

S_PER_TOK = float(os.environ.get("MOCK_S_PER_TOK", "0.002"))
DECODE_TPS = float(os.environ.get("MOCK_DECODE_TPS", "200"))
SERIALIZE = os.environ.get("MOCK_SERIALIZE", "1") == "1"
API_KEY = os.environ.get("MOCK_API_KEY")
FAULT_CONCURRENT = os.environ.get("MOCK_FAULT_CONCURRENT") == "1"

ENGINE_LOCK = threading.Lock()
WORDS = ("lattice ember quartz orbit fjord copper meadow signal crux "
         "harbor tundra velvet cinder pylon raster glyph").split()
_active = [0]
_counter_lock = threading.Lock()


def fake_tokens(prompt, n):
    seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "big")
    out = []
    for i in range(n):
        out.append(WORDS[(seed >> (i % 48)) % len(WORDS)] + " ")
        seed = (seed * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _sse(self, data):
        chunk = ("data: %s\n\n" % data).encode()
        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
        self.wfile.flush()

    def do_GET(self):
        if self.path.split("?")[0] in ("/health", "/"):
            body = b"ok"
        elif self.path.split("?")[0] == "/v1/models":
            body = json.dumps({"object": "list",
                               "data": [{"id": "mock-olmo-3-7b", "object": "model"}]}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?")[0] not in ("/v1/chat/completions", "/v1/completions"):
            self.send_error(404)
            return
        if API_KEY and self.headers.get("Authorization") != "Bearer " + API_KEY:
            self.send_error(401)
            return
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msgs = req.get("messages") or [{"content": req.get("prompt", "")}]
        prompt = " ".join(m.get("content", "") for m in msgs)
        prompt_tokens = max(1, len(prompt) // 4)
        max_tokens = int(req.get("max_tokens", 32))

        with _counter_lock:
            _active[0] += 1
            concurrent_entry = _active[0] >= 2

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            lock = ENGINE_LOCK if SERIALIZE else threading.Lock()
            with lock:
                # simulated prefill with keep-alive comments every ~0.5s
                remaining = prompt_tokens * S_PER_TOK
                while remaining > 0:
                    step = min(0.5, remaining)
                    time.sleep(step)
                    remaining -= step
                    if remaining > 0:
                        ka = b": keep-alive\n\n"
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(ka), ka))
                        self.wfile.flush()
                toks = fake_tokens(prompt, max_tokens)
                with _counter_lock:
                    faulty = FAULT_CONCURRENT and (concurrent_entry or _active[0] >= 2)
                if faulty:
                    toks[len(toks) // 2] = "CORRUPTED "
                n_reason = min(6, max_tokens // 4)
                for i, tok in enumerate(toks):
                    time.sleep(1.0 / DECODE_TPS)
                    last = i == len(toks) - 1
                    field = "reasoning" if i < n_reason else "content"
                    obj = {"id": "mock-1", "object": "chat.completion.chunk",
                           "model": "mock-olmo-3-7b",
                           "choices": [{"index": 0,
                                        "delta": {"role": "assistant", field: tok},
                                        "finish_reason": "stop" if last else None}]}
                    if last:
                        obj["usage"] = {"prompt_tokens": prompt_tokens,
                                        "completion_tokens": max_tokens,
                                        "total_tokens": prompt_tokens + max_tokens}
                    self._sse(json.dumps(obj))
            self._sse("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        finally:
            with _counter_lock:
                _active[0] -= 1


if __name__ == "__main__":
    port = int(os.environ.get("MOCK_PORT", "18080"))
    print("mock ATLAS api on :%d  serialize=%s  s/tok=%.4f  decode_tps=%.0f"
          % (port, SERIALIZE, S_PER_TOK, DECODE_TPS))
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
