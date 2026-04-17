/*
 * kernels/matmul.cu — Tiled GEMM for atlas-tensor
 *
 * Compiled by atlas-tensor/build.rs (nvcc, no Rust crates).
 * Called from Rust via:  extern "C" { fn atlas_matmul_f32(...) }
 *
 * Target: sm_75 (Tesla T4) and newer.
 * Tile size: 32×32 — fills a T4 SM (64 warps × 32 threads).
 *
 * Performance target (T4, f32):
 *   M=N=K=4096  →  ~4 TFLOPS  (T4 peak f32 = 8.1 TFLOPS)
 *
 * API:
 *   atlas_matmul_f32(A, B, C, M, N, K)   — C = A[M,K] × B[K,N]  (C zeroed on entry)
 *   atlas_matmul_f32_batched(A, B, C, M, N, K, batch) — batched version
 *   atlas_vec_add_f32(a, b, out, n)       — out = a + b (element-wise)
 *   atlas_scale_f32(x, s, out, n)         — out = x * s
 *   atlas_relu_f32(x, out, n)             — out[i] = max(0, x[i])
 *   atlas_softmax_f32(x, out, rows, cols) — softmax along last dim
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define TILE 32
#define MIN(a,b) ((a)<(b)?(a):(b))

/* ─── Tiled GEMM kernel ──────────────────────────────────────────────────── */

__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K
) {
    __shared__ float sA[TILE][TILE + 1]; /* +1 avoids bank conflicts */
    __shared__ float sB[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        /* Load tile of A */
        int aCol = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K)
            ? A[row * K + aCol] : 0.0f;

        /* Load tile of B */
        int bRow = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N)
            ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        /* Accumulate dot product for this tile */
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc = __fmaf_rn(sA[threadIdx.y][k], sB[k][threadIdx.x], acc);
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

/* ─── Element-wise kernels ───────────────────────────────────────────────── */

__global__ void vec_add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void scale_kernel(const float* x, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * s;
}

__global__ void relu_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

/* Online softmax (numerically stable) — one block per row */
__global__ void softmax_kernel(const float* x, float* out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* xr = x   + row * cols;
    float*       or_ = out + row * cols;

    /* Pass 1: find max */
    float mx = -1e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        mx = fmaxf(mx, xr[j]);
    /* warp-reduce max */
    for (int d = 16; d > 0; d >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
    mx = __shfl_sync(0xffffffff, mx, 0);

    /* Pass 2: sum of exp */
    float s = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        s += expf(xr[j] - mx);
    for (int d = 16; d > 0; d >>= 1)
        s += __shfl_down_sync(0xffffffff, s, d);
    s = __shfl_sync(0xffffffff, s, 0);

    /* Pass 3: write normalised values */
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        or_[j] = expf(xr[j] - mx) / s;
}

/* ─── AdamW moment update kernel ─────────────────────────────────────────── */
/* Used by atlas-optim via FFI */
__global__ void adamw_step_kernel(
    float* param,         /* updated in-place */
    float* m,             /* 1st moment, updated in-place */
    float* v,             /* 2nd moment, updated in-place */
    const float* grad,
    float lr, float beta1, float beta2,
    float eps, float weight_decay,
    float bc1, float bc2, /* bias-correction: 1/(1-beta^t) */
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g  = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    float m_hat = mi * bc1;
    float v_hat = vi * bc2;
    param[i] = param[i] * (1.0f - lr * weight_decay)
             - lr * m_hat / (sqrtf(v_hat) + eps);
}

/* ─── INT8 quantisation kernels ──────────────────────────────────────────── */

/* Quantise rows of f32 to INT8. scale[row] = max(|x|)/127. */
__global__ void quantize_int8_kernel(
    const float* __restrict__ input,
    int8_t*      __restrict__ output,
    float*       __restrict__ scale,
    int n_rows, int n_cols
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    /* Find row max */
    float mx = 0.0f;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        mx = fmaxf(mx, fabsf(input[row * n_cols + j]));
    for (int d = 16; d > 0; d >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
    mx = __shfl_sync(0xffffffff, mx, 0);

    if (threadIdx.x == 0) scale[row] = mx / 127.0f;
    __syncthreads();

    float s = scale[row];
    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;

    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
        float val = input[row * n_cols + j] * inv_s;
        /* clamp to [-127, 127] */
        val = fminf(fmaxf(val, -127.0f), 127.0f);
        output[row * n_cols + j] = (int8_t)__float2int_rn(val);
    }
}

__global__ void dequantize_int8_kernel(
    const int8_t* __restrict__ input,
    const float*  __restrict__ scale,
    float*        __restrict__ output,
    int n_rows, int n_cols
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    float s = scale[row];
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        output[row * n_cols + j] = (float)input[row * n_cols + j] * s;
}

/* INT4 — pack two 4-bit values per byte (upper nibble = even, lower = odd) */
__global__ void quantize_int4_kernel(
    const float* __restrict__ input,
    uint8_t*     __restrict__ output, /* n/2 bytes */
    float*       __restrict__ scale,
    int n
) {
    /* Single-block: find global max, then quantise pairs */
    float mx = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        mx = fmaxf(mx, fabsf(input[i]));
    for (int d = 16; d > 0; d >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, d));
    mx = __shfl_sync(0xffffffff, mx, 0);

    if (threadIdx.x == 0) *scale = mx / 7.0f; /* 4-bit signed: [-7, 7] */
    __syncthreads();

    float s = *scale;
    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;

    for (int i = threadIdx.x * 2; i < n - 1; i += blockDim.x * 2) {
        int q0 = __float2int_rn(fminf(fmaxf(input[i]   * inv_s, -7.0f), 7.0f));
        int q1 = __float2int_rn(fminf(fmaxf(input[i+1] * inv_s, -7.0f), 7.0f));
        /* Pack: high nibble = q0 & 0xF, low nibble = q1 & 0xF */
        output[i / 2] = (uint8_t)(((q0 & 0xF) << 4) | (q1 & 0xF));
    }
}

/* ─── RMSNorm kernel ─────────────────────────────────────────────────────── */
/* x[n], w[n] → out[n]   out[i] = x[i] * w[i] / sqrt(mean(x^2) + eps) */
__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ out,
    int n, float eps
) {
    /* General rmsnorm reduction: works for any n (n<32, n=32, n>32).
     *
     * Uses the correct per-warp active mask for __shfl_down_sync so that
     * partial warps (blockDim.x < 32) don't cause undefined behaviour.
     * Inter-warp reduction via smem[0] broadcast (no cross-warp shfl). */
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += x[i] * x[i];

    int warp_id   = threadIdx.x / 32;
    int warp_lane = threadIdx.x % 32;
    int n_warps   = (blockDim.x + 31) / 32;   /* ceil */

    /* Active mask for THIS warp: threads [warp_id*32 .. min(blockDim, warp_id*32+32)) */
    int warp_n  = min(32, blockDim.x - warp_id * 32);
    unsigned active = (warp_n >= 32) ? 0xffffffffu : ((1u << warp_n) - 1u);

    /* Warp-level reduce */
    for (int d = warp_n / 2; d > 0; d >>= 1)
        sum += __shfl_down_sync(active, sum, d);

    /* Warp lane-0 writes partial sum to smem */
    extern __shared__ float smem[];
    if (warp_lane == 0) smem[warp_id] = sum;
    __syncthreads();

    /* Thread 0 accumulates all warps then broadcasts via smem[0] */
    if (threadIdx.x == 0) {
        for (int w = 1; w < n_warps; w++)
            smem[0] += smem[w];
    }
    __syncthreads();

    float rms_inv = rsqrtf(smem[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = x[i] * rms_inv * w[i];
}

__global__ void rope_apply_kernel(
    float* __restrict__ x,
    int pos, int head_dim, float theta_base
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    if (i >= half) return;
    float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)head_dim);
    float angle = (float)pos * freq;
    float c = cosf(angle);
    float s = sinf(angle);
    float x0 = x[i];
    float x1 = x[i + half];
    x[i]        = x0 * c - x1 * s;
    x[i + half] = x0 * s + x1 * c;
}

/* ─── Fused SwiGLU: silu(gate) * up ─────────────────────────────────────── */
__global__ void silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float silu_g = g / (1.0f + expf(-g));   /* silu(x) = x * sigmoid(x) */
    out[i] = silu_g * up[i];
}

/* ─── BF16 conversion helper ──────────────────────────────────────────────── */
/* BF16 is exactly the upper 16 bits of IEEE 754 f32.                         */
/* Conversion: f32 = __uint_as_float((uint32_t)bf16 << 16)                    */
/* No cuda_bf16.h required — portable across sm_75+.                          */

static __device__ __forceinline__ float bf16u_to_f32(uint16_t h) {
    return __uint_as_float(((uint32_t)h) << 16);
}

/* ─── Efficient GEMV kernels for autoregressive inference (N=1) ─────────── */
/*                                                                            */
/* The tiled GEMM above is designed for large batch/prefill (N >> 1).        */
/* For N=1 (one token at a time, the typical autoregressive decode step),    */
/* one warp per row is optimal:                                               */
/*   - All 32 threads stride over K, accumulate, then warp-reduce to lane 0. */
/*   - Fully coalesced global memory access.                                  */
/*   - x vector (~16 KB for d=4096) fits in L1/L2 cache and is shared.      */

#define GEMV_ROWS_PER_BLOCK 8   /* warps (rows) per thread block */

/* F32 GEMV: y[M] = A[M×K] (f32) × x[K] (f32) */
__global__ void sgemv_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float*       __restrict__ y,
    int M, int K
) {
    int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= M) return;

    const float* A_row = A + (ptrdiff_t)row * K;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += 32)
        acc += A_row[k] * __ldg(x + k);   /* __ldg: load via read-only cache */

    /* Warp reduction: sum all 32 lanes into lane 0 */
    for (int d = 16; d > 0; d >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, d);

    if (threadIdx.x == 0) y[row] = acc;
}

/* BF16 weight × F32 activation GEMV: y[M] = A[M×K] (bf16) × x[K] (f32) */
__global__ void sgemv_bf16_kernel(
    const uint16_t* __restrict__ A,
    const float*    __restrict__ x,
    float*          __restrict__ y,
    int M, int K
) {
    int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= M) return;

    const uint16_t* A_row = A + (ptrdiff_t)row * K;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += 32)
        acc += bf16u_to_f32(A_row[k]) * __ldg(x + k);

    for (int d = 16; d > 0; d >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, d);

    if (threadIdx.x == 0) y[row] = acc;
}

/* ─── BF16 weight × F32 activation tiled GEMM (W16A32) ──────────────────── */
/*                                                                            */
/* A_bf16[M×K] in BF16 (uint16_t) × B[K×N] in F32 → C[M×N] in F32.         */
/* Memory savings vs F32: weights use ½ VRAM (14 GB instead of 28 GB for 7B) */

__global__ void sgemm_bf16_kernel(
    const uint16_t* __restrict__ A_bf16,   /* weights  [M × K] in BF16 */
    const float*    __restrict__ B,         /* input    [K × N] in F32  */
    float*          __restrict__ C,         /* output   [M × N] in F32  */
    int M, int N, int K
) {
    __shared__ float sA[TILE][TILE + 1]; /* +1 avoids bank conflicts */
    __shared__ float sB[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        /* Load tile of A: convert BF16 → F32 on the fly (no precision loss
         * for BF16-origin weights; upper 16 bits of f32 = bf16 pattern) */
        int aCol = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K)
            ? bf16u_to_f32(A_bf16[row * K + aCol]) : 0.0f;

        /* Load tile of B (already F32) */
        int bRow = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N)
            ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc = __fmaf_rn(sA[threadIdx.y][k], sB[k][threadIdx.x], acc);
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

/* ─── VRAM→VRAM memcpy helper ────────────────────────────────────────────── */
/* For KV cache writes that stay in VRAM */
__global__ void copy_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

/* ─── C-callable host wrappers ───────────────────────────────────────────── */

extern "C" {

void atlas_matmul_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    if (N == 1) {
        /* Efficient GEMV path for autoregressive single-token decoding.
         * ~32× fewer wasted FMAs vs tiled GEMM for N=1. */
        dim3 block(32, GEMV_ROWS_PER_BLOCK);
        dim3 grid((M + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
        sgemv_f32_kernel<<<grid, block>>>(A, B, C, M, K);
    } else {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
    }
    cudaDeviceSynchronize();
}

void atlas_vec_add_f32(const float* a, const float* b, float* out, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads>>>(a, b, out, n);
    cudaDeviceSynchronize();
}

void atlas_scale_f32(const float* x, float s, float* out, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(x, s, out, n);
    cudaDeviceSynchronize();
}

void atlas_relu_f32(const float* x, float* out, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(x, out, n);
    cudaDeviceSynchronize();
}

void atlas_softmax_f32(const float* x, float* out, int rows, int cols) {
    /* One block per row, 32 threads (one warp) */
    softmax_kernel<<<rows, 32>>>(x, out, rows, cols);
    cudaDeviceSynchronize();
}

void atlas_adamw_step(
    float* param, float* m, float* v, const float* grad,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int n
) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    adamw_step_kernel<<<blocks, threads>>>(param, m, v, grad,
        lr, beta1, beta2, eps, wd, bc1, bc2, n);
    cudaDeviceSynchronize();
}

void atlas_quantize_int8(
    const float* input, int8_t* output, float* scale,
    int n_rows, int n_cols
) {
    quantize_int8_kernel<<<n_rows, 32>>>(input, output, scale, n_rows, n_cols);
    cudaDeviceSynchronize();
}

void atlas_dequantize_int8(
    const int8_t* input, const float* scale, float* output,
    int n_rows, int n_cols
) {
    dequantize_int8_kernel<<<n_rows, 32>>>(input, scale, output, n_rows, n_cols);
    cudaDeviceSynchronize();
}

void atlas_quantize_int4(const float* input, uint8_t* output, float* scale, int n) {
    quantize_int4_kernel<<<1, 32>>>(input, output, scale, n);
    cudaDeviceSynchronize();
}

int atlas_cuda_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

/* Returns 1 if CUDA is available and working, 0 otherwise */
int atlas_cuda_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

void atlas_rmsnorm_f32(
    const float* x, const float* w, float* out,
    int n, float eps
) {
    int threads = MIN(256, n);
    int n_warps = (threads + 31) / 32;
    int smem    = n_warps * (int)sizeof(float);
    rmsnorm_kernel<<<1, threads, smem>>>(x, w, out, n, eps);
    cudaDeviceSynchronize();
}

void atlas_rope_apply_f32(float* x, int pos, int head_dim, float theta_base) {
    int half = head_dim / 2;
    int threads = MIN(256, half);
    int blocks  = (half + threads - 1) / threads;
    rope_apply_kernel<<<blocks, threads>>>(x, pos, head_dim, theta_base);
    cudaDeviceSynchronize();
}

void atlas_silu_mul_f32(const float* gate, const float* up, float* out, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads>>>(gate, up, out, n);
    cudaDeviceSynchronize();
}

void atlas_vram_copy_f32(const float* src, float* dst, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    copy_kernel<<<blocks, threads>>>(src, dst, n);
    cudaDeviceSynchronize();
}

/* BF16 weight × F32 activation GEMM / GEMV (W16A32).
 * Dispatches to efficient GEMV kernel for N=1 (autoregressive decoding)
 * or tiled GEMM for N>1 (prefill/batch). */
void atlas_sgemm_bf16_f32(
    const uint16_t* A_bf16, const float* B, float* C,
    int M, int N, int K
) {
    if (N == 1) {
        /* BF16 GEMV: one warp per row, coalesced loads, warp-reduce.
         * ~32× fewer wasted FMAs vs tiled GEMM for N=1. */
        dim3 block(32, GEMV_ROWS_PER_BLOCK);
        dim3 grid((M + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
        sgemv_bf16_kernel<<<grid, block>>>(A_bf16, B, C, M, K);
    } else {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        sgemm_bf16_kernel<<<grid, block>>>(A_bf16, B, C, M, N, K);
    }
    cudaDeviceSynchronize();
}

} /* extern "C" */
