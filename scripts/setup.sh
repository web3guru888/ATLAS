#!/bin/bash
# scripts/setup.sh — ATLAS development environment setup
#
# Checks prerequisites and prepares the workspace for development.
# Run once before starting Stage 1.

set -e

echo "=== ATLAS Setup ==="
echo ""

# 1. Rust toolchain
echo "[1/5] Checking Rust toolchain..."
if ! command -v rustc &>/dev/null; then
    echo "ERROR: rustc not found. Install from https://rustup.rs/"
    exit 1
fi
RUST_VER=$(rustc --version)
echo "  OK: $RUST_VER"
echo "  Minimum required: 1.75"

# 2. CUDA / nvcc
echo ""
echo "[2/5] Checking CUDA..."
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
    echo "  OK: nvcc $NVCC_VER"
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    echo "  GPU: $GPU"
else
    echo "  WARNING: nvcc not found — building CPU-only (no GPU acceleration)"
    echo "  Install CUDA 12.x from https://developer.nvidia.com/cuda-downloads"
fi

# 3. Check kernel directory
echo ""
echo "[3/5] Checking kernel files..."
for k in matmul.cu attention.cu quant.cu; do
    if [[ -f "kernels/$k" ]]; then
        echo "  OK: kernels/$k"
    else
        echo "  MISSING: kernels/$k"
    fi
done

# 4. Cargo workspace check
echo ""
echo "[4/5] Checking Cargo workspace..."
CRATE_COUNT=$(cargo metadata --no-deps --format-version 1 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['packages']))" 2>/dev/null \
    || echo "?")
echo "  Workspace crates: $CRATE_COUNT (expected: 17)"

# 5. Build Stage 1 (atlas-core + atlas-tensor)
echo ""
echo "[5/5] Building Stage 1 crates (atlas-core, atlas-tensor, atlas-grad)..."
if cargo build -p atlas-core -p atlas-tensor -p atlas-grad 2>&1 | tail -3; then
    echo "  Stage 1: BUILD OK"
else
    echo "  Stage 1: BUILD FAILED — check error above"
    exit 1
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Run tests:     cargo test -p atlas-core -p atlas-tensor -p atlas-grad"
echo "  2. Start Stage 1: implement atlas-optim (AdamW) and atlas-quant"
echo "  3. See CHARTER.md for full 7-stage build order"
echo ""
echo "  atlas train     — (Stage 6, not yet implemented)"
echo "  atlas discover  — (Stage 5, not yet implemented)"
