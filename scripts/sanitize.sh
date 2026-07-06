#!/usr/bin/env bash
# Usage: scripts/sanitize.sh <crate> [tool]   tool in: memcheck|racecheck|initcheck|synccheck
# Runs NVIDIA compute-sanitizer over a crate's test binary.
set -euo pipefail
CRATE="${1:?usage: sanitize.sh <crate> [tool]}"; TOOL="${2:-memcheck}"
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
# Build test binary without running
BIN=$(cargo test -p "$CRATE" --no-run --message-format=json 2>/dev/null \
      | python3 -c 'import json,sys
for l in sys.stdin:
    try: d=json.loads(l)
    except Exception: continue
    if d.get("profile",{}).get("test") and d.get("executable"): print(d["executable"])' | tail -1)
[ -n "$BIN" ] || { echo "no test binary found for $CRATE" >&2; exit 2; }
echo "sanitizing: $BIN ($TOOL)" >&2
compute-sanitizer --tool "$TOOL" --error-exitcode 1 "$BIN" --test-threads=1
