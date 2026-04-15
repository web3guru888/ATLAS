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

/* ─── C-callable host wrappers ───────────────────────────────────────────── */

extern "C" {

void atlas_matmul_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
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
    dequantize_int8_kernel<<<n_rows, 32>>>(input, output, scale, n_rows, n_cols);
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

} /* extern "C" */
