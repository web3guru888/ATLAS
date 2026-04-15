/*
 * kernels/matmul.cu — Tiled matrix multiplication kernel
 *
 * atlas-tensor calls this via raw extern "C" FFI — no cudarc, no tch.
 * Linked by atlas-tensor/build.rs.
 *
 * Atlas Philosophy: zero Rust crate dependencies.
 * CUDA is a system runtime, not a crate. We call it directly.
 *
 * Stage 1 target: f32 matmul on RTX 3090/4090 (sm_86).
 * Milestone: match cuBLAS throughput within 2× for M=N=K=4096.
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define TILE 16

/*
 * atlas_matmul_f32 — C = A × B
 *
 * A: [M, K] row-major f32
 * B: [K, N] row-major f32
 * C: [M, N] row-major f32 (output, zeroed by caller)
 */
extern "C" void atlas_matmul_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K
) {
    // TODO Stage 1: implement tiled GEMM kernel
    // For now: naive O(n³) reference implementation for correctness testing
    // Replace with tiled shared-memory GEMM (~400 LOC) once tests pass.
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

/*
 * Tiled GEMM (Stage 1 — to replace naive above)
 *
 * __global__ void atlas_matmul_tiled(
 *     const float* A, const float* B, float* C,
 *     int M, int N, int K
 * ) {
 *     __shared__ float sA[TILE][TILE];
 *     __shared__ float sB[TILE][TILE];
 *     int row = blockIdx.y * TILE + threadIdx.y;
 *     int col = blockIdx.x * TILE + threadIdx.x;
 *     float acc = 0.0f;
 *     for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
 *         sA[threadIdx.y][threadIdx.x] = (row < M && t*TILE+threadIdx.x < K)
 *             ? A[row * K + t*TILE + threadIdx.x] : 0.0f;
 *         sB[threadIdx.y][threadIdx.x] = (col < N && t*TILE+threadIdx.y < K)
 *             ? B[(t*TILE+threadIdx.y) * N + col] : 0.0f;
 *         __syncthreads();
 *         for (int k = 0; k < TILE; k++) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
 *         __syncthreads();
 *     }
 *     if (row < M && col < N) C[row * N + col] = acc;
 * }
 */
