# BF16 KV Cache — 32K Context Memory Budget (#24)

The GPU KV cache can now store keys/values in **BF16** instead of FP32, halving
KV VRAM so a 32K context fits alongside the model weights on the A100-40GB.

Enable at runtime: **`ATLAS_KV_BF16=1`** (default off → FP32 KV, unchanged).
Implementation: `GpuKvCache::new_bf16` + BF16 CUDA kernels
(`kv_cache_write_bf16`, `decode_attention_bf16`, `kv_range_to_bf16`). K/V are
computed in FP32 and stored BF16; attention up-converts to FP32 in registers —
**accumulation stays FP32, only storage precision drops** (same W16A32 pattern
the weights already use).

## KV bytes per token
`2 (K+V) × n_layers × n_kv_heads × head_dim × bytes_per_elem`

OLMo-3-32B-Think (W4): `n_layers=64, n_kv_heads=8, head_dim=128`
- FP32: `2·64·8·128·4  = 524,288 B/token = 0.500 MiB/token`
- BF16: `2·64·8·128·2  = 262,144 B/token = 0.250 MiB/token`

## VRAM budget on the A100-40GB (batch 1)
| Context | FP32 KV | BF16 KV | Weights (W4) | Total (BF16 KV) |
|--------:|--------:|--------:|-------------:|----------------:|
| 16K     |  8.0 GiB | 4.0 GiB | ~19.6 GiB   | ~23.6 GiB + act |
| 32K     | 16.0 GiB | 8.0 GiB | ~19.6 GiB   | ~27.6 GiB + act |
| 64K     | 32.0 GiB | 16.0 GiB| ~19.6 GiB   | ~35.6 GiB + act |

- **32K with FP32 KV = 19.6 + 16.0 = 35.6 GiB before activations → does not fit** with headroom.
- **32K with BF16 KV = 19.6 + 8.0 = 27.6 GiB + activations (~1–2 GiB) → ~29–30 GiB, comfortably inside 40 GiB.**
- Observed production steady-state today is ~27.2 GiB @ 16K FP32 KV (19.6 weights + 8.0 KV − shared) — so **32K BF16 KV ≈ the current 16K FP32 footprint**, i.e. the current production allocation already proves the 32K BF16 budget.

## Correctness
`bf16_kv_decode_matches_f32_within_tol` (atlas-tensor): identical K/V written to
an FP32 cache and a BF16 cache; same query; decode outputs compared.
Measured **max abs diff = 1.55e-4** at pos=40 (GQA 40 Q / 8 KV heads, head_dim=16
synthetic) — well within a BF16-aware tolerance. BF16 has 8 mantissa bits
(≈0.4% relative); softmax + FP32 accumulation keep the decode output
well-conditioned.

## Remaining validation (gated on an announced 32B maintenance window)
- Live NIAH sweep to 32K on the 32B endpoint (`/opt/atlas-tools/niah.py --json`).
- Greedy-parity spot check FP32-KV vs BF16-KV on the live model.
- Long-context prefill pairs with #22 (batched prefill, merged) and benefits
  from the #23 decode/BF16-GEMM work to keep 32K TTFT usable.

These require loading the 32B alongside the running server (would exceed 40 GiB),
so they run in a maintenance window, not against the live serving instance.
