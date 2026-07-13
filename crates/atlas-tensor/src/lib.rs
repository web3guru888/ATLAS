//! atlas-tensor — The seed of everything.
//!
//! Pure Rust f32 tensor with optional CUDA acceleration.
//! Zero external Rust crate dependencies.
//!
//! When compiled with CUDA (`atlas_cuda` cfg flag set by build.rs):
//!   - GPU tensors hold a device pointer in `gpu_ptr`
//!   - Ops dispatch to `extern "C"` functions in `kernels/matmul.cu`
//!
//! When compiled CPU-only (`atlas_cpu_only`):
//!   - All ops are pure Rust, no unsafe

use atlas_core::{AtlasError, Device, DType, Result};

// ── CUDA FFI declarations ──────────────────────────────────────────────────
// Only compiled in when build.rs successfully compiled the CUDA kernels.
#[cfg(atlas_cuda)]
mod ffi {
    use std::ffi::c_int;
    extern "C" {
        pub fn atlas_matmul_f32(
            a: *const f32, b: *const f32, c: *mut f32,
            m: c_int, n: c_int, k: c_int,
        );
        pub fn atlas_vec_add_f32(a: *const f32, b: *const f32, out: *mut f32, n: c_int);
        pub fn atlas_scale_f32(x: *const f32, s: f32, out: *mut f32, n: c_int);
        pub fn atlas_relu_f32(x: *const f32, out: *mut f32, n: c_int);
        pub fn atlas_softmax_f32(x: *const f32, out: *mut f32, rows: c_int, cols: c_int);
        pub fn atlas_cuda_available() -> c_int;
        pub fn atlas_cuda_device_count() -> c_int;
        pub fn atlas_rmsnorm_f32(
            x: *const f32, w: *const f32, out: *mut f32, n: c_int, eps: f32,
        );
        pub fn atlas_rope_apply_f32(x: *mut f32, pos: c_int, head_dim: c_int, theta_base: f32);
        pub fn atlas_silu_mul_f32(gate: *const f32, up: *const f32, out: *mut f32, n: c_int);
        pub fn atlas_vram_copy_f32(src: *const f32, dst: *mut f32, n: c_int);
        /// AdamW step on HOST buffers (staged through device memory in C).
        /// Returns 0 on success, non-zero CUDA error code on failure.
        pub fn atlas_adamw_step(
            param: *mut f32, m: *mut f32, v: *mut f32, grad: *const f32,
            lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
            bc1: f32, bc2: f32, n: c_int,
        ) -> c_int;
        /// BF16 weight × F32 activation GEMM (W16A32).
        /// A_bf16[M,K] stored as uint16_t (BF16 bit pattern), B[K,N] F32, C[M,N] F32.
        pub fn atlas_sgemm_bf16_f32(
            a_bf16: *const u16, b: *const f32, c: *mut f32,
            m: c_int, n: c_int, k: c_int,
        );
        /// W4A32 group-quantized GEMV (weight-only int4, symmetric).
        /// packed[M, K/8] uint32 (compressed-tensors pack-quantized layout),
        /// scales[M, K/G] BF16 bit patterns, x[K] F32 → y[M] F32.
        pub fn atlas_gemv_w4_f32(
            packed: *const u32, scales: *const u16,
            x: *const f32, y: *mut f32,
            m: c_int, k: c_int, g: c_int,
        );
        /// W4 → F32 dequantize into a VRAM scratch buffer (#22 Step 2).
        /// packed[M, K/8] uint32, scales[M, K/G] BF16 bits → out[M, K] F32.
        pub fn atlas_dequant_w4_f32(
            packed: *const u32, scales: *const u16, out: *mut f32,
            m: c_int, k: c_int, g: c_int,
        );
        /// Explicitly drain all pending GPU work (cudaDeviceSynchronize).
        /// Rarely needed — use only when CPU must read GPU output without going
        /// through GpuVec::download() (which syncs implicitly via cudaMemcpy D2H).
        pub fn atlas_sync();

        // ── GPU Decode Attention (Issue #18) ─────────────────────────────────
        /// Per-head RMSNorm in-place: OLMo-2/3 QK-norm before RoPE.
        /// x: [n_heads × head_dim], w: [n_heads × head_dim] (per-head weights).
        pub fn atlas_qk_norm_inplace(
            x: *mut f32, w: *const f32,
            n_heads: c_int, head_dim: c_int, eps: f32,
        );
        /// Apply precomputed YaRN RoPE cos/sin tables to Q or K in-place.
        /// x:       [n_heads × head_dim] in VRAM (f32)
        /// cos_tab: [max_seq × half_dim] in VRAM
        /// sin_tab: [max_seq × half_dim] in VRAM
        pub fn atlas_rope_precomputed(
            x: *mut f32,
            cos_tab: *const f32, sin_tab: *const f32,
            n_heads: c_int, head_dim: c_int, pos: c_int, max_seq: c_int,
        );
        /// Write new K,V into GPU KV cache at position `pos`.
        pub fn atlas_kv_cache_write(
            k_cache: *mut f32, v_cache: *mut f32,
            new_k: *const f32, new_v: *const f32,
            pos: c_int, n_kv_heads: c_int, head_dim: c_int,
        );
        /// Grouped-query decode attention: no Q,K,V PCIe transfers.
        /// q:       [n_heads × head_dim]
        /// k_cache: [max_seq × n_kv_heads × head_dim]
        /// v_cache: [max_seq × n_kv_heads × head_dim]
        /// out:     [n_heads × head_dim]
        pub fn atlas_decode_attention(
            q: *const f32,
            k_cache: *const f32, v_cache: *const f32,
            out: *mut f32,
            n_heads: c_int, n_kv_heads: c_int, head_dim: c_int,
            pos: c_int, scale: f32, window_size: c_int,
        );
        /// BF16 KV cache write (#24): F32 K/V in, BF16 (u16) stored.
        pub fn atlas_kv_cache_write_bf16(
            k_cache: *mut u16, v_cache: *mut u16,
            new_k: *const f32, new_v: *const f32,
            pos: c_int, n_kv_heads: c_int, head_dim: c_int,
        );
        /// Convert-copy `n` F32 values into a BF16 destination (#24 chunked write_range).
        pub fn atlas_kv_range_to_bf16(src: *const f32, dst: *mut u16, n: c_int);
        /// Grouped-query decode attention against a BF16 KV cache (#24).
        pub fn atlas_decode_attention_bf16(
            q: *const f32,
            k_cache: *const u16, v_cache: *const u16,
            out: *mut f32,
            n_heads: c_int, n_kv_heads: c_int, head_dim: c_int,
            pos: c_int, scale: f32, window_size: c_int,
        );
        /// GPU argmax: finds the index of the maximum element in x[0..n].
        /// Both x and out_idx must be device pointers.
        /// GPU argmax: writes index as f32 to out_f32 (index < 2^24 is exact).
        pub fn atlas_gpu_argmax(x: *const f32, out_f32: *mut f32, n: c_int);
        pub fn atlas_take_kernel_error() -> c_int;
    }
}

/// Rust-side sticky error flag for GPU operations that fail outside a CUDA
/// kernel launch (e.g. a D2H memcpy error in `GpuVec::download`).
#[cfg(atlas_cuda)]
static RUST_GPU_ERROR: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

#[cfg(atlas_cuda)]
pub(crate) fn note_rust_gpu_error(code: i32) {
    let _ = RUST_GPU_ERROR.compare_exchange(
        0, code,
        std::sync::atomic::Ordering::Relaxed,
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Return and clear the sticky GPU error flag (0 = no error since last call).
///
/// Covers (a) failed async kernel launches recorded by the C wrappers
/// (`atlas_note_launch_err`) and (b) failed device↔host copies recorded on
/// the Rust side. Inference callers check this at end-of-token: a non-zero
/// value means the token's output is untrustworthy (a kernel silently did
/// not run — e.g. launch failure under VRAM pressure) and the CPU path must
/// recompute it. Always 0 on CPU-only builds.
pub fn take_kernel_error() -> i32 {
    #[cfg(atlas_cuda)]
    {
        let c = unsafe { ffi::atlas_take_kernel_error() };
        let r = RUST_GPU_ERROR.swap(0, std::sync::atomic::Ordering::Relaxed);
        if c != 0 { c } else { r }
    }
    #[cfg(not(atlas_cuda))]
    0
}

/// Drain all pending GPU work (cudaDeviceSynchronize).
/// Rarely needed in normal use — GpuVec::download() syncs implicitly.
/// Call this when you need to measure GPU timing or confirm all kernels have finished.
pub fn device_sync() {
    #[cfg(atlas_cuda)]
    unsafe { ffi::atlas_sync() }
}

/// GPU argmax: returns the index of the maximum element in a GPU vector.
///
/// Returns  when CUDA is unavailable or the vector is not GPU-resident.
/// This is 10× faster than downloading all logits to CPU for greedy decoding
/// (transfers only 4 bytes instead of vocab_size × 4 bytes ≈ 400 KB for OLMo-3).
pub fn gpu_argmax(x: &GpuVec) -> Option<u32> {
    #[cfg(atlas_cuda)]
    if let Some(xp) = x.gpu_ptr() {
        // Result stored as f32: indices < 2^24 (~16M) are exact in f32, vocab is ~100K
        if let Some(out_buf) = gpu::GpuBuf::alloc(1) {
            unsafe {
                ffi::atlas_gpu_argmax(xp, out_buf.ptr, x.len as i32);
            }
            // Download 4 bytes (float representation of the int index)
            let floats = out_buf.download();
            if !floats.is_empty() {
                return Some(floats[0] as u32);
            }
        }
    }
    // CPU fallback
    x.cpu.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
}

/// Returns true if CUDA was compiled in AND a device is reachable at runtime.
pub fn cuda_available() -> bool {
    #[cfg(atlas_cuda)]
    unsafe { ffi::atlas_cuda_available() != 0 }
    #[cfg(not(atlas_cuda))]
    false
}

/// Returns number of CUDA devices (0 if CUDA not compiled or no devices).
pub fn cuda_device_count() -> i32 {
    #[cfg(atlas_cuda)]
    unsafe { ffi::atlas_cuda_device_count() }
    #[cfg(not(atlas_cuda))]
    0
}

// ── GPU memory helpers ─────────────────────────────────────────────────────
// A GpuBuffer wraps a raw CUDA device pointer with RAII dealloc.

#[cfg(atlas_cuda)]
mod gpu {
    use std::ffi::c_void;

    extern "C" {
        fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        fn cudaFree(ptr: *mut c_void) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
        /// Async allocation on stream `stream` (pass null for the default stream).
        /// Does NOT block the host. Available since CUDA 11.2.
        fn cudaMallocAsync(ptr: *mut *mut c_void, size: usize, stream: *mut c_void) -> i32;
        /// Async free on stream `stream`. Memory is returned to the pool after
        /// all preceding operations on `stream` complete.
        fn cudaFreeAsync(ptr: *mut c_void, stream: *mut c_void) -> i32;
    }

    const CUDA_MEMCPY_H2D: i32 = 1;
    const CUDA_MEMCPY_D2H: i32 = 2;

    /// Activation scratch buffer for inference temporaries (Q, K, V, gate, etc.).
    ///
    /// # Allocation strategy
    ///
    /// We try `cudaMallocAsync` first (CUDA 11.2+, our server has 13.0).
    /// Unlike `cudaMalloc`, async allocation does NOT synchronize with the device —
    /// it submits the allocation into stream-0 ordering without blocking the CPU.
    /// This allows consecutive kernel launches to proceed without implicit sync
    /// barriers between them, which is the primary driver of throughput for decode.
    ///
    /// Fallback: if `cudaMallocAsync` returns an error (shouldn't on CUDA 11.2+),
    /// we fall back to synchronous `cudaMalloc`.
    pub struct GpuBuf {
        pub ptr: *mut f32,
        pub len: usize,
        /// True if allocated with cudaMallocAsync; drop uses cudaFreeAsync.
        async_alloc: bool,
    }
    unsafe impl Send for GpuBuf {}
    unsafe impl Sync for GpuBuf {}
    impl std::fmt::Debug for GpuBuf {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "GpuBuf {{ ptr: {:?}, len: {} }}", self.ptr, self.len)
        }
    }

    impl GpuBuf {
        pub fn alloc(len: usize) -> Option<Self> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // Try async allocation on stream-0 (null = default stream).
            // cudaMallocAsync does not synchronize — no implicit device barrier.
            let async_err = unsafe {
                cudaMallocAsync(&mut ptr, len * 4, std::ptr::null_mut())
            };
            if async_err == 0 {
                return Some(Self { ptr: ptr as *mut f32, len, async_alloc: true });
            }
            // Fall back to synchronous cudaMalloc (CUDA < 11.2 or pool exhausted).
            let err = unsafe { cudaMalloc(&mut ptr, len * 4) };
            if err != 0 { return None; }
            Some(Self { ptr: ptr as *mut f32, len, async_alloc: false })
        }
        pub fn upload(data: &[f32]) -> Option<Self> {
            let buf = Self::alloc(data.len())?;
            let err = unsafe {
                cudaMemcpy(buf.ptr as *mut c_void, data.as_ptr() as *const c_void,
                           data.len() * 4, CUDA_MEMCPY_H2D)
            };
            if err != 0 { return None; }
            Some(buf)
        }
        pub fn download(&self) -> Vec<f32> {
            let mut out = vec![0.0f32; self.len];
            let err = unsafe {
                cudaMemcpy(out.as_mut_ptr() as *mut c_void, self.ptr as *const c_void,
                           self.len * 4, CUDA_MEMCPY_D2H)
            };
            if err != 0 {
                // Do NOT silently return zeros — record the failure so
                // take_kernel_error() lets the caller recompute on CPU.
                crate::note_rust_gpu_error(err);
            }
            out
        }
    }
    impl Drop for GpuBuf {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                if self.async_alloc {
                    // Async free: memory returned to pool after preceding stream-0 work.
                    unsafe { cudaFreeAsync(self.ptr as *mut c_void, std::ptr::null_mut()); }
                } else {
                    unsafe { cudaFree(self.ptr as *mut c_void); }
                }
            }
        }
    }

    /// GPU buffer for BF16 weights (u16, 2 bytes/element).
    /// Model weights are loaded once and live in VRAM for the process lifetime —
    /// no need for async alloc here; sync cudaMalloc is fine at load time.
    pub struct GpuBufBf16 {
        pub ptr: *mut u16,
        pub len: usize,
    }
    unsafe impl Send for GpuBufBf16 {}
    unsafe impl Sync for GpuBufBf16 {}
    impl std::fmt::Debug for GpuBufBf16 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "GpuBufBf16 {{ ptr: {:?}, len: {} }}", self.ptr, self.len)
        }
    }
    impl GpuBufBf16 {
        pub fn alloc(len: usize) -> Option<Self> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let err = unsafe { cudaMalloc(&mut ptr, len * 2) }; // 2 bytes per BF16
            if err != 0 { return None; }
            Some(Self { ptr: ptr as *mut u16, len })
        }
        pub fn upload(data: &[u16]) -> Option<Self> {
            let buf = Self::alloc(data.len())?;
            let err = unsafe {
                cudaMemcpy(
                    buf.ptr as *mut c_void,
                    data.as_ptr() as *const c_void,
                    data.len() * 2, // 2 bytes per u16
                    CUDA_MEMCPY_H2D,
                )
            };
            if err != 0 { return None; }
            Some(buf)
        }
        /// Download BF16 device buffer to a host `Vec<u16>` (bit patterns).
        pub fn download(&self) -> Vec<u16> {
            let mut out = vec![0u16; self.len];
            let err = unsafe {
                cudaMemcpy(out.as_mut_ptr() as *mut c_void, self.ptr as *const c_void,
                           self.len * 2, CUDA_MEMCPY_D2H)
            };
            if err != 0 { crate::note_rust_gpu_error(err); }
            out
        }
    }
    impl Drop for GpuBufBf16 {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { cudaFree(self.ptr as *mut c_void); }
            }
        }
    }

    /// Raw u32 GPU buffer — used for packed int4 weights (8 nibbles / u32).
    pub struct GpuBufU32 {
        pub ptr: *mut u32,
        pub len: usize,
    }
    unsafe impl Send for GpuBufU32 {}
    unsafe impl Sync for GpuBufU32 {}
    impl std::fmt::Debug for GpuBufU32 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "GpuBufU32 {{ ptr: {:?}, len: {} }}", self.ptr, self.len)
        }
    }
    impl GpuBufU32 {
        pub fn alloc(len: usize) -> Option<Self> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let err = unsafe { cudaMalloc(&mut ptr, len * 4) }; // 4 bytes per u32
            if err != 0 { return None; }
            Some(Self { ptr: ptr as *mut u32, len })
        }
        pub fn upload(data: &[u32]) -> Option<Self> {
            let buf = Self::alloc(data.len())?;
            let err = unsafe {
                cudaMemcpy(
                    buf.ptr as *mut c_void,
                    data.as_ptr() as *const c_void,
                    data.len() * 4,
                    CUDA_MEMCPY_H2D,
                )
            };
            if err != 0 { return None; }
            Some(buf)
        }
    }
    impl Drop for GpuBufU32 {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { cudaFree(self.ptr as *mut c_void); }
            }
        }
    }

    /// Discriminated union: f32, BF16, or W4 (int4 group-quantized) weights.
    /// `GpuMatrix` holds `Option<GpuBufKind>` so it can store any precision.
    pub enum GpuBufKind {
        /// Full f32 weights in VRAM (4 bytes/element).
        F32(GpuBuf),
        /// BF16 weights in VRAM (2 bytes/element).  W16A32: activations stay f32.
        BF16(GpuBufBf16),
        /// Weight-only int4 (compressed-tensors pack-quantized, symmetric,
        /// group scales).  W4A32: ~0.56 bytes/element incl. scales.
        W4 {
            /// [rows × cols/8] packed nibbles as u32 (little-endian nibbles).
            packed: GpuBufU32,
            /// [rows × cols/group] per-group scales, BF16 bit patterns.
            scales: GpuBufBf16,
            /// Quantization group size along the input (cols) dimension.
            group: usize,
        },
    }
    impl GpuBufKind {
        pub fn is_bf16(&self) -> bool { matches!(self, GpuBufKind::BF16(_)) }
        pub fn is_w4(&self)   -> bool { matches!(self, GpuBufKind::W4 { .. }) }
    }
}


// ── GpuMatrix — weight matrix pinned in VRAM ──────────────────────────────

/// A matrix pre-uploaded to GPU VRAM (upload once, multiply many times).
///
/// Supports two precisions:
///   - **F32** (4 bytes/elem): default for small models or CPU-only builds.
///   - **BF16** (2 bytes/elem): for large BF16 models (e.g. OLMo-3-7B: 14 GB vs 28 GB).
///     Uses W16A32 arithmetic — weights in BF16, activations in F32.
///     Conversion BF16→F32 is done inline in the CUDA kernel (no precision loss for
///     weights that were originally stored as BF16).
///
/// On CPU-only builds (no `atlas_cuda` cfg): this is a zero-overhead no-op;
/// `sgemm()` always returns `false` and the caller uses its own CPU path.
pub struct GpuMatrix {
    #[cfg(atlas_cuda)]
    buf: Option<gpu::GpuBufKind>,
    pub rows: usize,   // output dimension
    pub cols: usize,   // input dimension
}

impl GpuMatrix {
    /// Upload a row-major **f32** matrix [rows × cols] to GPU VRAM.
    /// Falls back gracefully if CUDA is not available.
    pub fn upload(data: &[f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self {
            #[cfg(atlas_cuda)]
            buf: if cuda_available() {
                gpu::GpuBuf::upload(data).map(gpu::GpuBufKind::F32)
            } else { None },
            rows,
            cols,
        }
    }

    /// Upload a row-major **BF16** matrix [rows × cols] to GPU VRAM.
    ///
    /// `data` contains BF16 bit patterns as `u16` (native BF16 representation).
    /// Uses W16A32: weights in BF16 VRAM, activations remain f32.
    /// Falls back gracefully if CUDA is not available.
    pub fn upload_bf16(data: &[u16], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self {
            #[cfg(atlas_cuda)]
            buf: if cuda_available() {
                gpu::GpuBufBf16::upload(data).map(gpu::GpuBufKind::BF16)
            } else { None },
            rows,
            cols,
        }
    }

    /// Upload a W4 (int4 group-quantized) matrix to GPU VRAM.
    ///
    /// `packed`: [rows × cols/8] u32 words, compressed-tensors pack-quantized
    /// layout (col j → word j/8, little-endian nibble j%8, nibble = q+8).
    /// `scales_bf16`: [rows × cols/group] BF16 bit patterns.
    /// W4A32: dequantization happens inline in the CUDA kernel.
    /// Falls back gracefully (CPU-only matrix) if CUDA is not available.
    pub fn upload_w4(packed: &[u32], scales_bf16: &[u16], rows: usize, cols: usize, group: usize) -> Self {
        debug_assert_eq!(cols % 8, 0);
        debug_assert_eq!(group % 8, 0);
        debug_assert_eq!(packed.len(), rows * (cols / 8));
        debug_assert_eq!(scales_bf16.len(), rows * (cols / group));
        Self {
            #[cfg(atlas_cuda)]
            buf: if cuda_available() {
                match (gpu::GpuBufU32::upload(packed), gpu::GpuBufBf16::upload(scales_bf16)) {
                    (Some(p), Some(s)) => Some(gpu::GpuBufKind::W4 { packed: p, scales: s, group }),
                    _ => None,
                }
            } else { None },
            rows,
            cols,
        }
    }

    /// Create a placeholder matrix with no storage (neither GPU nor CPU).
    /// Used by `OlmoModel::new_uninit` so that constructing a large model
    /// before loading real weights does not allocate hundreds of GB.
    pub fn empty(rows: usize, cols: usize) -> Self {
        Self {
            #[cfg(atlas_cuda)]
            buf: None,
            rows,
            cols,
        }
    }

    /// Whether the matrix is resident in GPU VRAM (any precision).
    pub fn is_on_gpu(&self) -> bool {
        #[cfg(atlas_cuda)]
        { self.buf.is_some() }
        #[cfg(not(atlas_cuda))]
        { false }
    }

    /// Whether the matrix uses BF16 precision in VRAM.
    pub fn is_bf16(&self) -> bool {
        #[cfg(atlas_cuda)]
        { self.buf.as_ref().map_or(false, |b| b.is_bf16()) }
        #[cfg(not(atlas_cuda))]
        { false }
    }

    /// Whether the matrix uses W4 (int4 group-quantized) precision in VRAM.
    pub fn is_w4(&self) -> bool {
        #[cfg(atlas_cuda)]
        { self.buf.as_ref().map_or(false, |b| b.is_w4()) }
        #[cfg(not(atlas_cuda))]
        { false }
    }

    /// GPU SGEMM: `out[m × n] = self[m × k] × rhs[k × n]` (row-major).
    ///
    /// Weight matrix is already in VRAM. Only `rhs` (the input activations)
    /// is uploaded per call — typically a tiny x-vector (k floats).
    /// Dispatches to the BF16 or F32 kernel based on the stored weight precision.
    ///
    /// Returns `true` if GPU was used; caller should fall back to CPU if `false`.
    pub fn sgemm(&self, rhs: &[f32], k: usize, n: usize, out: &mut [f32]) -> bool {
        let m = self.rows;
        debug_assert_eq!(self.cols, k);
        debug_assert_eq!(rhs.len(), k * n);
        debug_assert_eq!(out.len(), m * n);
        #[cfg(atlas_cuda)]
        if let Some(ref a_buf) = self.buf {
            if let Some(b_buf) = gpu::GpuBuf::upload(rhs) {
                if let Some(c_buf) = gpu::GpuBuf::alloc(m * n) {
                    // Isolate this launch so the post-launch check below is
                    // specific to it (any prior error was someone else's to
                    // handle — sgemm's contract is bool success).
                    let _ = take_kernel_error();
                    match a_buf {
                        gpu::GpuBufKind::F32(f32_buf) => unsafe {
                            ffi::atlas_matmul_f32(
                                f32_buf.ptr, b_buf.ptr, c_buf.ptr,
                                m as i32, n as i32, k as i32,
                            );
                        },
                        gpu::GpuBufKind::BF16(bf16_buf) => unsafe {
                            ffi::atlas_sgemm_bf16_f32(
                                bf16_buf.ptr, b_buf.ptr, c_buf.ptr,
                                m as i32, n as i32, k as i32,
                            );
                        },
                        gpu::GpuBufKind::W4 { packed, scales, group } => unsafe {
                            if n == 1 {
                                ffi::atlas_gemv_w4_f32(
                                    packed.ptr, scales.ptr, b_buf.ptr, c_buf.ptr,
                                    m as i32, k as i32, *group as i32,
                                );
                            } else {
                                // #22 Step 2: dequant → f32 scratch → GEMM.
                                let Some(guard) = w4_scratch_at_least(m * k)
                                    else { return false; };
                                let scratch = guard.as_ref().unwrap();
                                ffi::atlas_dequant_w4_f32(
                                    packed.ptr, scales.ptr, scratch.ptr,
                                    m as i32, k as i32, *group as i32,
                                );
                                ffi::atlas_matmul_f32(
                                    scratch.ptr, b_buf.ptr, c_buf.ptr,
                                    m as i32, n as i32, k as i32,
                                );
                            }
                        },
                    }
                    let result = c_buf.download();
                    if take_kernel_error() != 0 {
                        // Kernel launch or copy failed (e.g. VRAM pressure):
                        // `result` is uninitialized garbage. Report failure so
                        // the caller runs its CPU fallback instead of
                        // silently corrupting the forward pass.
                        return false;
                    }
                    out.copy_from_slice(&result);
                    return true;
                }
            }
        }
        false
    }

    /// GPU SGEMM where the input is already in VRAM (no H2D upload needed).
    ///
    /// Returns the output `GpuVec` (still in VRAM — no D2H download).
    /// This is the zero-copy path: input stays in VRAM between operations.
    /// Dispatches to BF16 or F32 kernel based on weight precision.
    ///
    /// Falls back to None if CUDA not available (caller must use CPU path).
    pub fn sgemm_vec(&self, x: &GpuVec, n: usize) -> Option<GpuVec> {
        let m = self.rows;
        #[cfg(atlas_cuda)]
        if let (Some(ref a_buf), Some(ref x_buf)) = (&self.buf, &x.buf) {
            let out_buf = gpu::GpuBuf::alloc(m * n)?;
            match a_buf {
                gpu::GpuBufKind::F32(f32_buf) => unsafe {
                    ffi::atlas_matmul_f32(
                        f32_buf.ptr, x_buf.ptr, out_buf.ptr,
                        m as i32, n as i32, self.cols as i32,
                    );
                },
                gpu::GpuBufKind::BF16(bf16_buf) => unsafe {
                    ffi::atlas_sgemm_bf16_f32(
                        bf16_buf.ptr, x_buf.ptr, out_buf.ptr,
                        m as i32, n as i32, self.cols as i32,
                    );
                },
                gpu::GpuBufKind::W4 { packed, scales, group } => unsafe {
                    if n == 1 {
                        ffi::atlas_gemv_w4_f32(
                            packed.ptr, scales.ptr, x_buf.ptr, out_buf.ptr,
                            m as i32, self.cols as i32, *group as i32,
                        );
                    } else {
                        // #22 Step 2: dequant → f32 scratch → GEMM.
                        let guard = w4_scratch_at_least(m * self.cols)?;
                        let scratch = guard.as_ref().unwrap();
                        ffi::atlas_dequant_w4_f32(
                            packed.ptr, scales.ptr, scratch.ptr,
                            m as i32, self.cols as i32, *group as i32,
                        );
                        ffi::atlas_matmul_f32(
                            scratch.ptr, x_buf.ptr, out_buf.ptr,
                            m as i32, n as i32, self.cols as i32,
                        );
                    }
                },
            }
            return Some(GpuVec {
                buf: Some(out_buf),
                cpu: vec![0.0f32; m * n],
                len: m * n,
            });
        }
        None
    }
}

/// Reusable VRAM scratch for the W4 batch-GEMM path (#22 Step 2).
///
/// W4 weights are stored packed (int4). For batched activations (n > 1) the
/// GEMV kernel would re-stream the packed weights once per token, so instead
/// the whole Linear is dequantized into this f32 scratch once per call and
/// the existing cuBLAS GEMM consumes it (weight traffic ≈ 8.6 B/element per
/// chunk vs 0.56 B × n for GEMV — break-even n ≈ 15, ~17× less at n = 256).
///
/// Grow-only and process-global: the largest 32B Linear is 27,648 × 5,120
/// f32 ≈ 566 MB, allocated once and reused for every layer / chunk.
/// Reuse is safe without extra synchronization: all launches go to stream-0,
/// so the next dequant cannot overtake a previous GEMM that reads the
/// scratch. The Mutex only serializes CPU-side access to the allocation.
#[cfg(atlas_cuda)]
static W4_GEMM_SCRATCH: std::sync::Mutex<Option<gpu::GpuBuf>> =
    std::sync::Mutex::new(None);

/// Lock the W4 dequant scratch, growing it to ≥ `len` f32 elements.
/// Returns `None` if VRAM allocation fails (caller falls back to CPU).
#[cfg(atlas_cuda)]
fn w4_scratch_at_least(
    len: usize,
) -> Option<std::sync::MutexGuard<'static, Option<gpu::GpuBuf>>> {
    let mut guard = match W4_GEMM_SCRATCH.lock() {
        Ok(g) => g,
        // The scratch holds no cross-call invariants — a panic elsewhere
        // cannot corrupt it, so a poisoned lock is safe to take over.
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.as_ref().map_or(true, |b| b.len < len) {
        *guard = None; // free the old buffer before allocating the bigger one
        *guard = Some(gpu::GpuBuf::alloc(len)?);
    }
    Some(guard)
}

impl std::fmt::Debug for GpuMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(atlas_cuda)]
        let dtype = match &self.buf {
            Some(gpu::GpuBufKind::F32(_))   => "f32",
            Some(gpu::GpuBufKind::BF16(_))  => "bf16",
            Some(gpu::GpuBufKind::W4 {..})  => "w4",
            None => "cpu",
        };
        #[cfg(not(atlas_cuda))]
        let dtype = "cpu";
        write!(f, "GpuMatrix({}×{}, {})", self.rows, self.cols, dtype)
    }
}

// ── GpuVec — mutable GPU buffer for activation tensors ────────────────────

/// A mutable GPU buffer for transient activations (hidden states, KV cache).
///
/// Unlike `GpuMatrix` (static weight matrix pre-pinned in VRAM),
/// `GpuVec` is created and destroyed per-forward-pass for intermediate results.
///
/// On CPU-only builds, this is a zero-cost wrapper around `Vec<f32>`.
pub struct GpuVec {
    #[cfg(atlas_cuda)]
    buf: Option<gpu::GpuBuf>,
    /// CPU fallback storage.
    pub cpu: Vec<f32>,
    /// Number of f32 elements.
    pub len: usize,
}

impl GpuVec {
    /// Create a GPU vector filled with zeros.
    pub fn zeros(len: usize) -> Self {
        let cpu = vec![0.0f32; len];
        Self {
            // Upload the zero-filled host buffer instead of a bare alloc:
            // cudaMalloc/cudaMallocAsync do NOT guarantee zeroed memory
            // (async pool allocations recycle prior contents). Found via
            // compute-sanitizer, which poisons fresh allocations.
            #[cfg(atlas_cuda)]
            buf: if cuda_available() { gpu::GpuBuf::upload(&cpu) } else { None },
            cpu,
            len,
        }
    }

    /// Upload a CPU slice to GPU (copy also kept in cpu for fallback).
    pub fn from_slice(data: &[f32]) -> Self {
        Self {
            #[cfg(atlas_cuda)]
            buf: if cuda_available() { gpu::GpuBuf::upload(data) } else { None },
            cpu: data.to_vec(),
            len: data.len(),
        }
    }

    /// Download from GPU to CPU Vec<f32>.
    pub fn download(&self) -> Vec<f32> {
        #[cfg(atlas_cuda)]
        if let Some(ref b) = self.buf { return b.download(); }
        self.cpu.clone()
    }

    /// Duplicate this vector **without leaving the device**.
    ///
    /// On GPU: allocates a fresh buffer and runs the device-to-device
    /// `copy_kernel` — zero PCIe traffic, no synchronisation (stream-ordered).
    /// On CPU (or CPU-only builds): clones the host data.
    ///
    /// Callers previously wrote `GpuVec::from_slice(&x.download())` to get a
    /// scratch copy of a resident activation — a full D2H + H2D round-trip
    /// plus an implicit sync, per call. The 2026-07-06 nsys baseline showed
    /// PCIe staging at 79% of GPU time on small-model inference; the pre-norm
    /// transformer path did two such round-trips per layer per token.
    pub fn dup(&self) -> GpuVec {
        #[cfg(atlas_cuda)]
        if let Some(ref src) = self.buf {
            if let Some(dst) = gpu::GpuBuf::alloc(self.len) {
                unsafe { ffi::atlas_vram_copy_f32(src.ptr, dst.ptr, self.len as i32); }
                return GpuVec {
                    buf: Some(dst),
                    cpu: vec![0.0f32; self.len],
                    len: self.len,
                };
            }
        }
        GpuVec::from_slice(&self.download())
    }

    /// Whether the data is resident in VRAM.
    pub fn is_on_gpu(&self) -> bool {
        #[cfg(atlas_cuda)]
        { self.buf.is_some() }
        #[cfg(not(atlas_cuda))]
        { false }
    }

    /// In-place element-wise add from another GpuVec.
    pub fn add_inplace(&mut self, other: &GpuVec) {
        debug_assert_eq!(self.len, other.len);
        #[cfg(atlas_cuda)]
        if let (Some(ref a), Some(ref b)) = (&self.buf, &other.buf) {
            if let Some(out) = gpu::GpuBuf::alloc(self.len) {
                unsafe { ffi::atlas_vec_add_f32(a.ptr, b.ptr, out.ptr, self.len as i32); }
                let mut new_buf = Some(out);
                core::mem::swap(&mut self.buf, &mut new_buf);
                return;
            }
        }
        // CPU fallback — never trust the host shadows of GPU-resident
        // operands: once a vec lives in VRAM its `cpu` field is stale zeros.
        // download() returns the authoritative contents either way.
        let av = self.download();
        let bv = other.download();
        let sum: Vec<f32> = av.iter().zip(bv.iter()).map(|(x, y)| x + y).collect();
        *self = GpuVec::from_slice(&sum);
    }

    /// In-place RMSNorm: self = rmsnorm(self, w, eps).
    pub fn rmsnorm_inplace(&mut self, w: &GpuVec, eps: f32) {
        debug_assert_eq!(self.len, w.len);
        #[cfg(atlas_cuda)]
        if let (Some(ref x_buf), Some(ref w_buf)) = (&self.buf, &w.buf) {
            if let Some(out_buf) = gpu::GpuBuf::alloc(self.len) {
                unsafe {
                    ffi::atlas_rmsnorm_f32(
                        x_buf.ptr, w_buf.ptr, out_buf.ptr, self.len as i32, eps,
                    );
                }
                let mut new_buf = Some(out_buf);
                core::mem::swap(&mut self.buf, &mut new_buf);
                return;
            }
        }
        // CPU fallback — download authoritative contents (host shadows of
        // GPU-resident vecs are stale; norm weights ARE typically resident).
        let xv = self.download();
        let wv = w.download();
        let ss: f32 = xv.iter().map(|&v| v * v).sum::<f32>() / self.len as f32;
        let rms_inv = 1.0 / (ss + eps).sqrt();
        let out: Vec<f32> = xv.iter().zip(wv.iter()).map(|(x, w)| x * rms_inv * w).collect();
        *self = GpuVec::from_slice(&out);
    }

    /// Raw GPU pointer (for use in FFI calls). None on CPU-only builds.
    #[cfg(atlas_cuda)]
    pub fn gpu_ptr(&self) -> Option<*mut f32> {
        self.buf.as_ref().map(|b| b.ptr)
    }

    /// Mutable raw GPU pointer.
    #[cfg(atlas_cuda)]
    pub fn gpu_ptr_mut(&mut self) -> Option<*mut f32> {
        self.buf.as_mut().map(|b| b.ptr)
    }
}

impl std::fmt::Debug for GpuVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuVec(len={}, gpu={})", self.len, self.is_on_gpu())
    }
}

// ── Public GPU kernel wrappers ─────────────────────────────────────────────

/// Apply in-place RoPE rotation to a GpuVec representing one attention head.
///
/// `x` must have length `head_dim`. `pos` is the sequence position.
/// `theta_base` is the RoPE base frequency (default: 10_000.0 or 500_000.0).
pub fn rope_apply_gpu(x: &mut GpuVec, pos: usize, head_dim: usize, theta_base: f32) {
    #[cfg(atlas_cuda)]
    if let Some(ptr) = x.gpu_ptr_mut() {
        unsafe { ffi::atlas_rope_apply_f32(ptr, pos as i32, head_dim as i32, theta_base); }
        return;
    }
    // CPU fallback
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta_base.powf((2 * i) as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let (s, c) = angle.sin_cos();
        let x0 = x.cpu[i];
        let x1 = x.cpu[i + half];
        x.cpu[i]        = x0 * c - x1 * s;
        x.cpu[i + half] = x0 * s + x1 * c;
    }
}

/// Fused SwiGLU: `out[i] = silu(gate[i]) * up[i]`. Returns a new GpuVec.
pub fn silu_mul_gpu(gate: &GpuVec, up: &GpuVec) -> GpuVec {
    debug_assert_eq!(gate.len, up.len);
    let n = gate.len;
    #[cfg(atlas_cuda)]
    if let (Some(g_ptr), Some(u_ptr)) = (gate.gpu_ptr(), up.gpu_ptr()) {
        if let Some(out_buf) = gpu::GpuBuf::alloc(n) {
            unsafe { ffi::atlas_silu_mul_f32(g_ptr, u_ptr, out_buf.ptr, n as i32); }
            return GpuVec { buf: Some(out_buf), cpu: vec![0.0f32; n], len: n };
        }
    }
    // CPU fallback — download authoritative contents (host shadows of
    // GPU-resident vecs are stale zeros). from_slice retries the upload so
    // downstream GPU ops can continue when the failure was transient.
    let gv = gate.download();
    let uv = up.download();
    let cpu: Vec<f32> = gv.iter().zip(uv.iter())
        .map(|(&g, &u)| { let sg = g / (1.0 + (-g).exp()); sg * u })
        .collect();
    GpuVec::from_slice(&cpu)
}

/// Call the GPU AdamW step kernel for one parameter group.
///
/// `param`, `m`, `v` are HOST pointers, updated in-place. The C wrapper
/// stages them through device memory (H2D → kernel → D2H) and returns a
/// CUDA error code, which is propagated here as `Err`.
///
/// Returns `Ok(true)` if the GPU kernel ran successfully, `Ok(false)` if
/// CUDA is unavailable (caller should use the CPU path), and `Err` if the
/// kernel or a memcpy failed — in which case host buffers for this group
/// are unchanged (the D2H copy-back only happens after a successful kernel).
pub fn adamw_step_gpu(
    param: *mut f32, m: *mut f32, v: *mut f32, grad: *const f32,
    lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
    bc1: f32, bc2: f32, n: usize,
) -> Result<bool> {
    #[cfg(atlas_cuda)]
    if cuda_available() {
        let code = unsafe {
            ffi::atlas_adamw_step(param, m, v, grad, lr, beta1, beta2, eps, wd, bc1, bc2, n as i32)
        };
        if code != 0 {
            return Err(AtlasError::Other(format!(
                "CUDA adamw_step failed with error code {code} (see stderr for cudaGetErrorString)"
            )));
        }
        return Ok(true);
    }
    #[cfg(not(atlas_cuda))]
    {
        // Silence unused-parameter warnings on CPU-only builds.
        let _ = (param, m, v, grad, lr, beta1, beta2, eps, wd, bc1, bc2, n);
    }
    Ok(false)
}

/// Apply per-head RMSNorm in-place on Q or K (OLMo-2/3 QK-norm).
/// `x`: [n_heads × head_dim] in VRAM.
/// `w`: [n_heads × head_dim] norm weights in VRAM.
pub fn qk_norm_inplace_gpu(x: &mut GpuVec, w: &GpuVec, n_heads: usize, head_dim: usize, eps: f32) {
    #[cfg(atlas_cuda)]
    if let (Some(xp), Some(wp)) = (x.gpu_ptr_mut(), w.gpu_ptr()) {
        unsafe {
            ffi::atlas_qk_norm_inplace(xp, wp, n_heads as i32, head_dim as i32, eps);
        }
        return;
    }
    // CPU fallback: per-head RMSNorm. `x` is host-resident here (otherwise
    // the GPU path above ran), but `w` may live in VRAM — download it.
    let wv = w.download();
    let _n = n_heads * head_dim;
    for h in 0..n_heads {
        let s = h * head_dim;
        let xe = &mut x.cpu[s..s + head_dim];
        let we = &wv[s..s + head_dim];
        let rms_inv = (xe.iter().map(|&v| v * v).sum::<f32>() / head_dim as f32 + eps)
            .sqrt()
            .recip();
        for i in 0..head_dim {
            xe[i] = xe[i] * rms_inv * we[i];
        }
    }
}

// ── GPU KV Cache (decode attention without PCIe round-trips) ─────────────────

/// Precomputed RoPE cos/sin tables resident in VRAM.
///
/// Uploaded once at model load. Eliminates CPU-side RoPE and the
/// Q/K D2H + H2D round-trips that dominate decode latency.
pub struct GpuRopeTables {
    /// cos[pos × half_dim] — flat, row-major, in VRAM
    pub cos: GpuVec,
    /// sin[pos × half_dim] — flat, row-major, in VRAM
    pub sin: GpuVec,
    pub max_seq:  usize,
    pub half_dim: usize,
}

impl GpuRopeTables {
    /// Upload precomputed tables. `cos_flat` / `sin_flat` are each `max_seq × half_dim`.
    pub fn upload(cos_flat: &[f32], sin_flat: &[f32], max_seq: usize, half_dim: usize) -> Option<Self> {
        if !cuda_available() { return None; }
        let cos = GpuVec::from_slice(cos_flat);
        let sin = GpuVec::from_slice(sin_flat);
        if !cos.is_on_gpu() || !sin.is_on_gpu() {
            // Upload failed (VRAM pressure): returning Some would make
            // apply_to() silently skip RoPE for every token. Report failure
            // so the model uses the CPU-attention path instead.
            return None;
        }
        Some(Self { cos, sin, max_seq, half_dim })
    }

    /// Apply RoPE in-place to `x` (shape [n_heads × head_dim]) at position `pos`.
    pub fn apply_to(&self, x: &mut GpuVec, n_heads: usize, head_dim: usize, pos: usize) {
        #[cfg(atlas_cuda)]
        if let (Some(xp), Some(cp), Some(sp)) = (x.gpu_ptr_mut(), self.cos.gpu_ptr(), self.sin.gpu_ptr()) {
            unsafe {
                ffi::atlas_rope_precomputed(
                    xp, cp, sp,
                    n_heads as i32, head_dim as i32, pos as i32, self.max_seq as i32,
                );
            }
        }
        // CPU fallback: do nothing (caller handles CPU path separately)
    }
}

/// GPU-resident key-value cache for autoregressive decode.
///
/// Stores K and V for all past positions in VRAM. On A100 BF16:
///   OLMo-3-7B, 4096 max context: 32 × 8 × 128 × 4096 × 4 B × 2 = 268 MB
///
/// All cache reads/writes stay in VRAM — zero PCIe transfers for attention.
pub struct GpuKvCache {
    /// keys:   [max_seq × n_kv_heads × head_dim] f32 in VRAM (F32 mode).
    /// Empty placeholder when the cache is BF16 (see `keys_bf16`).
    pub keys:   GpuVec,
    /// values: [max_seq × n_kv_heads × head_dim] f32 in VRAM (F32 mode).
    pub values: GpuVec,
    pub max_seq:    usize,
    pub n_kv_heads: usize,
    pub head_dim:   usize,
    /// BF16 KV storage (#24): when `Some`, K/V live here as u16 (half the VRAM
    /// of the F32 path) and `keys`/`values` are unused. Enables 32K context on
    /// the A100-40GB. Selected via `new_bf16` (env `ATLAS_KV_BF16`).
    #[cfg(atlas_cuda)]
    keys_bf16:   Option<gpu::GpuBufBf16>,
    #[cfg(atlas_cuda)]
    values_bf16: Option<gpu::GpuBufBf16>,
}

impl GpuKvCache {
    /// Allocate a zeroed F32 GPU KV cache (default).
    pub fn new(max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Option<Self> {
        if !cuda_available() { return None; }
        let total = max_seq * n_kv_heads * head_dim;
        let keys   = GpuVec::zeros(total);
        let values = GpuVec::zeros(total);
        Some(Self {
            keys, values, max_seq, n_kv_heads, head_dim,
            #[cfg(atlas_cuda)] keys_bf16:   None,
            #[cfg(atlas_cuda)] values_bf16: None,
        })
    }

    /// Allocate a **BF16** GPU KV cache (#24) — half the VRAM of `new`, so a
    /// 32K context fits alongside BF16/W4 weights on the A100-40GB. K/V are
    /// computed in F32 and stored BF16; attention up-converts to F32 for the
    /// dot products (accumulation stays F32 — only storage precision drops).
    pub fn new_bf16(max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Option<Self> {
        #[cfg(atlas_cuda)]
        {
            if !cuda_available() { return None; }
            let total = max_seq * n_kv_heads * head_dim;
            let keys_bf16   = gpu::GpuBufBf16::upload(&vec![0u16; total])?;
            let values_bf16 = gpu::GpuBufBf16::upload(&vec![0u16; total])?;
            return Some(Self {
                keys:   GpuVec::zeros(0),
                values: GpuVec::zeros(0),
                max_seq, n_kv_heads, head_dim,
                keys_bf16:   Some(keys_bf16),
                values_bf16: Some(values_bf16),
            });
        }
        #[cfg(not(atlas_cuda))]
        { let _ = (max_seq, n_kv_heads, head_dim); None }
    }

    /// True when this cache stores K/V in BF16.
    pub fn is_bf16(&self) -> bool {
        #[cfg(atlas_cuda)]
        { return self.keys_bf16.is_some(); }
        #[cfg(not(atlas_cuda))]
        { false }
    }

    /// Whether the KV data is resident in VRAM (either precision).
    pub fn is_on_gpu(&self) -> bool {
        #[cfg(atlas_cuda)]
        { if self.keys_bf16.is_some() { return true; } }
        self.keys.is_on_gpu()
    }

    /// Download the key cache to host `Vec<f32>` (up-converts from BF16 if needed).
    pub fn download_keys(&self) -> Vec<f32> {
        #[cfg(atlas_cuda)]
        if let Some(kb) = self.keys_bf16.as_ref() {
            return kb.download().iter().map(|&u| f32::from_bits((u as u32) << 16)).collect();
        }
        self.keys.download()
    }

    /// Download the value cache to host `Vec<f32>` (up-converts from BF16 if needed).
    pub fn download_values(&self) -> Vec<f32> {
        #[cfg(atlas_cuda)]
        if let Some(vb) = self.values_bf16.as_ref() {
            return vb.download().iter().map(|&u| f32::from_bits((u as u32) << 16)).collect();
        }
        self.values.download()
    }

    /// Write new K and V vectors (size [n_kv_heads × head_dim]) at position `pos`.
    pub fn write(&mut self, pos: usize, new_k: &GpuVec, new_v: &GpuVec) {
        #[cfg(atlas_cuda)]
        {
            if let (Some(kb), Some(vb)) = (self.keys_bf16.as_ref(), self.values_bf16.as_ref()) {
                if let (Some(nkp), Some(nvp)) = (new_k.gpu_ptr(), new_v.gpu_ptr()) {
                    unsafe {
                        ffi::atlas_kv_cache_write_bf16(
                            kb.ptr, vb.ptr, nkp as *const f32, nvp as *const f32,
                            pos as i32, self.n_kv_heads as i32, self.head_dim as i32,
                        );
                    }
                }
                return;
            }
            if let (Some(kp), Some(vp), Some(nkp), Some(nvp)) = (
                self.keys.gpu_ptr_mut(), self.values.gpu_ptr_mut(),
                new_k.gpu_ptr(), new_v.gpu_ptr()
            ) {
                unsafe {
                    ffi::atlas_kv_cache_write(
                        kp, vp, nkp, nvp,
                        pos as i32, self.n_kv_heads as i32, self.head_dim as i32,
                    );
                }
            }
        }
        #[cfg(not(atlas_cuda))]
        { let _ = (pos, new_k, new_v); }
    }

    /// Write K/V for `m` consecutive positions starting at `start` in ONE
    /// device-to-device copy each (#22 chunked prefill — replaces `m`
    /// per-position `write` calls, each of which cost two H2D uploads).
    ///
    /// `k`/`v`: [m × n_kv_heads × head_dim] position-major, GPU-resident —
    /// exactly the cache's own layout, so the destination range
    /// `[start × kv_dim, (start+m) × kv_dim)` is contiguous.
    ///
    /// Returns `false` when the copy could not be performed (CPU-only build
    /// or non-resident buffers) — caller must treat the GPU cache as stale.
    pub fn write_range(&mut self, start: usize, m: usize, k: &GpuVec, v: &GpuVec) -> bool {
        let kv_dim = self.n_kv_heads * self.head_dim;
        debug_assert!(start + m <= self.max_seq);
        debug_assert_eq!(k.len, m * kv_dim);
        debug_assert_eq!(v.len, m * kv_dim);
        #[cfg(atlas_cuda)]
        {
            if let (Some(kb), Some(vb)) = (self.keys_bf16.as_ref(), self.values_bf16.as_ref()) {
                if let (Some(nkp), Some(nvp)) = (k.gpu_ptr(), v.gpu_ptr()) {
                    unsafe {
                        ffi::atlas_kv_range_to_bf16(nkp as *const f32, kb.ptr.add(start * kv_dim), (m * kv_dim) as i32);
                        ffi::atlas_kv_range_to_bf16(nvp as *const f32, vb.ptr.add(start * kv_dim), (m * kv_dim) as i32);
                    }
                    return true;
                }
                return false;
            }
            if let (Some(kp), Some(vp), Some(nkp), Some(nvp)) = (
                self.keys.gpu_ptr_mut(), self.values.gpu_ptr_mut(),
                k.gpu_ptr(), v.gpu_ptr()
            ) {
                unsafe {
                    ffi::atlas_vram_copy_f32(nkp, kp.add(start * kv_dim), (m * kv_dim) as i32);
                    ffi::atlas_vram_copy_f32(nvp, vp.add(start * kv_dim), (m * kv_dim) as i32);
                }
                return true;
            }
        }
        let _ = (start, m, k, v, kv_dim);
        false
    }

    /// Chunked-prefill attention (#22): decode attention for `m` consecutive
    /// query positions `start..start+m` with ONE Q upload and ONE output
    /// buffer (the per-position kernel is launched `m` times at pointer
    /// offsets — launches are cheap and async; the removed cost is the 2·m
    /// alloc+PCIe round-trips of calling `decode_attention` per token).
    ///
    /// `q`: [m × n_heads × head_dim] position-major, GPU-resident, RoPE'd.
    /// K/V for positions `< start + m` must already be in the cache
    /// (see [`Self::write_range`]) — token `s` attends to `0..=start+s`.
    ///
    /// Returns [m × n_heads × head_dim] in VRAM, or None if resources are
    /// unavailable (caller falls back to the CPU attention path).
    pub fn decode_attention_prefill(
        &self,
        q: &GpuVec,
        m: usize,
        n_heads: usize,
        start: usize,
        scale: f32,
        window_size: usize,
    ) -> Option<GpuVec> {
        let d = n_heads * self.head_dim;
        debug_assert_eq!(q.len, m * d);
        #[cfg(atlas_cuda)]
        if let (Some(kb), Some(vb)) = (self.keys_bf16.as_ref(), self.values_bf16.as_ref()) {
            if let Some(qp) = q.gpu_ptr() {
                if let Some(out_buf) = gpu::GpuBuf::alloc(m * d) {
                    for s in 0..m {
                        unsafe {
                            ffi::atlas_decode_attention_bf16(
                                qp.add(s * d), kb.ptr as *const u16, vb.ptr as *const u16,
                                out_buf.ptr.add(s * d),
                                n_heads as i32, self.n_kv_heads as i32, self.head_dim as i32,
                                (start + s) as i32, scale, window_size as i32,
                            );
                        }
                    }
                    return Some(GpuVec { buf: Some(out_buf), cpu: vec![0.0f32; m * d], len: m * d });
                }
            }
            return None;
        }
        #[cfg(atlas_cuda)]
        if let (Some(qp), Some(kp), Some(vp)) =
            (q.gpu_ptr(), self.keys.gpu_ptr(), self.values.gpu_ptr())
        {
            if let Some(out_buf) = gpu::GpuBuf::alloc(m * d) {
                for s in 0..m {
                    unsafe {
                        ffi::atlas_decode_attention(
                            qp.add(s * d), kp, vp, out_buf.ptr.add(s * d),
                            n_heads as i32, self.n_kv_heads as i32, self.head_dim as i32,
                            (start + s) as i32, scale, window_size as i32,
                        );
                    }
                }
                return Some(GpuVec {
                    buf: Some(out_buf),
                    cpu: vec![0.0f32; m * d],
                    len: m * d,
                });
            }
        }
        let _ = (q, m, n_heads, start, scale, window_size, d);
        None
    }

    /// Compute grouped-query decode attention. Returns output [n_heads × head_dim].
    ///
    /// `q`: [n_heads × head_dim] query vector (after RoPE).
    /// `n_heads`: total Q heads (n_kv_heads × group).
    /// `scale`: attention scale = 1/sqrt(head_dim) × attn_factor.
    /// `window_size`: 0 for full attention, >0 for sliding window.
    pub fn decode_attention(
        &self,
        q: &GpuVec,
        n_heads: usize,
        pos: usize,
        scale: f32,
        window_size: usize,
    ) -> Option<GpuVec> {
        #[cfg(atlas_cuda)]
        if let (Some(kb), Some(vb)) = (self.keys_bf16.as_ref(), self.values_bf16.as_ref()) {
            if let Some(qp) = q.gpu_ptr() {
                let out_len = n_heads * self.head_dim;
                if let Some(out_buf) = gpu::GpuBuf::alloc(out_len) {
                    unsafe {
                        ffi::atlas_decode_attention_bf16(
                            qp, kb.ptr as *const u16, vb.ptr as *const u16, out_buf.ptr,
                            n_heads as i32, self.n_kv_heads as i32, self.head_dim as i32,
                            pos as i32, scale, window_size as i32,
                        );
                    }
                    return Some(GpuVec { buf: Some(out_buf), cpu: vec![0.0f32; out_len], len: out_len });
                }
            }
            return None;
        }
        #[cfg(atlas_cuda)]
        if let (Some(qp), Some(kp), Some(vp)) = (q.gpu_ptr(), self.keys.gpu_ptr(), self.values.gpu_ptr()) {
            let out_len = n_heads * self.head_dim;
            if let Some(out_buf) = gpu::GpuBuf::alloc(out_len) {
                unsafe {
                    ffi::atlas_decode_attention(
                        qp, kp, vp, out_buf.ptr,
                        n_heads as i32, self.n_kv_heads as i32, self.head_dim as i32,
                        pos as i32, scale, window_size as i32,
                    );
                }
                return Some(GpuVec {
                    buf: Some(out_buf),
                    cpu: vec![0.0f32; out_len],
                    len: out_len,
                });
            }
        }
        None
    }

    /// Reset cache for a new sequence.
    ///
    /// The GPU buffers do NOT need to be zeroed: the decode attention kernel only
    /// reads positions `t ∈ [0, pos]`, and `pos` is managed by `OlmoModel::pos`
    /// which is reset to 0 in `OlmoModel::reset()`.  Stale data at positions
    /// beyond `pos` is never accessed.
    ///
    /// This is a NO-OP by design — any CPU zeroing of the GPU buffer would
    /// require 1 GB of memory allocation and PCIe traffic for a 32-layer model,
    /// which is ~100× more expensive than the benefit.
    pub fn reset(&mut self) {
        /* intentionally empty — see doc comment */
    }
}

// ── Tensor ─────────────────────────────────────────────────────────────────

/// A multi-dimensional f32 tensor.
///
/// On CPU: `data` holds the values directly.
/// On GPU: `data` holds a mirrored host copy; `gpu_ptr` is the device pointer.
#[derive(Debug)]
pub struct Tensor {
    data:    Vec<f32>,
    shape:   Vec<usize>,
    dtype:   DType,
    device:  Device,
    /// GPU buffer (Some when placed on CUDA device).
    #[cfg(atlas_cuda)]
    gpu_buf: Option<gpu::GpuBuf>,
}

// Manual Clone (GpuBuf is not Clone)
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            data:   self.data.clone(),
            shape:  self.shape.clone(),
            dtype:  self.dtype,
            device: self.device,
            #[cfg(atlas_cuda)]
            gpu_buf: None, // cloned tensor starts on CPU; call .to_cuda() if needed
        }
    }
}

impl Tensor {
    fn new_cpu(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data, shape, dtype: DType::F32, device: Device::Cpu,
            #[cfg(atlas_cuda)]
            gpu_buf: None,
        }
    }

    /// Create a zero-filled CPU tensor.
    pub fn zeros(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        Self::new_cpu(vec![0.0f32; n], shape.to_vec())
    }

    /// Create a tensor filled with a constant.
    pub fn full(shape: &[usize], value: f32) -> Self {
        let n = shape.iter().product();
        Self::new_cpu(vec![value; n], shape.to_vec())
    }

    /// Create from owned data.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(AtlasError::ShapeMismatch {
                expected: vec![expected],
                got:      vec![data.len()],
            });
        }
        Ok(Self::new_cpu(data, shape))
    }

    /// Upload to GPU. Returns self if CUDA not available (silently stays CPU).
    #[allow(unused_mut)]
    pub fn to_cuda(mut self) -> Self {
        #[cfg(atlas_cuda)]
        {
            if cuda_available() {
                if let Some(buf) = gpu::GpuBuf::upload(&self.data) {
                    self.device  = Device::Cuda(0);
                    self.gpu_buf = Some(buf);
                }
            }
        }
        self
    }

    /// Sync GPU→CPU (no-op if already on CPU).
    #[allow(unused_mut)]
    pub fn to_cpu(mut self) -> Self {
        #[cfg(atlas_cuda)]
        {
            if let Some(ref buf) = self.gpu_buf {
                self.data   = buf.download();
                self.device = Device::Cpu;
            }
            self.gpu_buf = None;
        }
        self
    }

    pub fn numel(&self)          -> usize        { self.shape.iter().product() }
    pub fn shape(&self)          -> &[usize]     { &self.shape }
    pub fn ndim(&self)           -> usize        { self.shape.len() }
    pub fn dtype(&self)          -> DType        { self.dtype }
    pub fn device(&self)         -> Device       { self.device }
    pub fn is_cuda(&self)        -> bool         { self.device != Device::Cpu }

    pub fn as_slice(&self) -> Result<&[f32]> {
        if self.is_cuda() {
            return Err(AtlasError::Other(
                "as_slice() on GPU tensor — call .to_cpu() first".into()));
        }
        Ok(&self.data)
    }

    pub fn as_slice_mut(&mut self) -> Result<&mut [f32]> {
        if self.is_cuda() {
            return Err(AtlasError::Other(
                "as_slice_mut() on GPU tensor".into()));
        }
        Ok(&mut self.data)
    }

    // ── Arithmetic ops ────────────────────────────────────────────────────

    /// Matrix multiply: [M,K] × [K,N] → [M,N]
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(AtlasError::Other("matmul requires 2D tensors".into()));
        }
        let (m, k, n) = (self.shape[0], self.shape[1], other.shape[1]);
        if k != other.shape[0] {
            return Err(AtlasError::ShapeMismatch {
                expected: vec![m, k],
                got:      vec![other.shape[0], n],
            });
        }

        #[cfg(atlas_cuda)]
        if self.is_cuda() && other.is_cuda() {
            if let (Some(a_buf), Some(b_buf)) = (&self.gpu_buf, &other.gpu_buf) {
                let mut out = Tensor::zeros(&[m, n]).to_cuda();
                if let Some(c_buf) = &out.gpu_buf {
                    unsafe {
                        ffi::atlas_matmul_f32(
                            a_buf.ptr, b_buf.ptr, c_buf.ptr,
                            m as i32, n as i32, k as i32,
                        );
                    }
                    out.data = c_buf.download();
                    return Ok(out);
                }
            }
        }

        // CPU fallback
        let a = self.as_slice()?;
        let b = other.as_slice()?;
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_ip = a[i * k + p];
                for j in 0..n {
                    out[i * n + j] += a_ip * b[p * n + j];
                }
            }
        }
        Tensor::from_vec(out, vec![m, n])
    }

    /// Element-wise add.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(AtlasError::ShapeMismatch {
                expected: self.shape.clone(),
                got:      other.shape.clone(),
            });
        }
        #[cfg(atlas_cuda)]
        if self.is_cuda() && other.is_cuda() {
            if let (Some(a), Some(b)) = (&self.gpu_buf, &other.gpu_buf) {
                let mut out = Tensor::zeros(&self.shape).to_cuda();
                if let Some(c) = &out.gpu_buf {
                    unsafe { ffi::atlas_vec_add_f32(a.ptr, b.ptr, c.ptr, self.numel() as i32); }
                    out.data = c.download();
                    return Ok(out);
                }
            }
        }
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a,b)| a+b).collect();
        Tensor::from_vec(data, self.shape.clone())
    }

    /// Element-wise multiply.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(AtlasError::ShapeMismatch {
                expected: self.shape.clone(),
                got:      other.shape.clone(),
            });
        }
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a,b)| a*b).collect();
        Tensor::from_vec(data, self.shape.clone())
    }

    /// Scalar multiply.
    pub fn scale(&self, s: f32) -> Tensor {
        #[cfg(atlas_cuda)]
        if self.is_cuda() {
            if let Some(buf) = &self.gpu_buf {
                let mut out = Tensor::zeros(&self.shape).to_cuda();
                if let Some(c) = &out.gpu_buf {
                    unsafe { ffi::atlas_scale_f32(buf.ptr, s, c.ptr, self.numel() as i32); }
                    out.data = c.download();
                    return out;
                }
            }
        }
        let data: Vec<f32> = self.data.iter().map(|x| x * s).collect();
        Tensor::new_cpu(data, self.shape.clone())
    }

    /// ReLU activation.
    pub fn relu(&self) -> Tensor {
        #[cfg(atlas_cuda)]
        if self.is_cuda() {
            if let Some(buf) = &self.gpu_buf {
                let mut out = Tensor::zeros(&self.shape).to_cuda();
                if let Some(c) = &out.gpu_buf {
                    unsafe { ffi::atlas_relu_f32(buf.ptr, c.ptr, self.numel() as i32); }
                    out.data = c.download();
                    return out;
                }
            }
        }
        let data: Vec<f32> = self.data.iter().map(|x| x.max(0.0)).collect();
        Tensor::new_cpu(data, self.shape.clone())
    }

    /// Softmax along the last dimension.
    pub fn softmax(&self) -> Result<Tensor> {
        let last = *self.shape.last()
            .ok_or_else(|| AtlasError::Other("softmax on 0-dim tensor".into()))?;
        let rows = self.numel() / last;

        #[cfg(atlas_cuda)]
        if self.is_cuda() {
            if let Some(buf) = &self.gpu_buf {
                let mut out = Tensor::zeros(&self.shape).to_cuda();
                if let Some(c) = &out.gpu_buf {
                    unsafe { ffi::atlas_softmax_f32(buf.ptr, c.ptr, rows as i32, last as i32); }
                    out.data = c.download();
                    return Ok(out);
                }
            }
        }
        let mut data = self.data.clone();
        for r in 0..rows {
            let row = &mut data[r * last..(r + 1) * last];
            let mx  = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            row.iter_mut().for_each(|x| *x = (*x - mx).exp());
            let s: f32 = row.iter().sum();
            row.iter_mut().for_each(|x| *x /= s);
        }
        Tensor::from_vec(data, self.shape.clone())
    }

    /// Reshape (same total elements).
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor> {
        if shape.iter().product::<usize>() != self.numel() {
            return Err(AtlasError::Other("reshape: element count mismatch".into()));
        }
        Ok(Tensor::new_cpu(self.data.clone(), shape))
    }

    /// Transpose 2D tensor.
    pub fn transpose(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(AtlasError::Other("transpose requires 2D".into()));
        }
        let (m, n) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                data[j * m + i] = self.data[i * n + j];
            }
        }
        Tensor::from_vec(data, vec![n, m])
    }

    /// Sum all elements.
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Mean of all elements.
    pub fn mean(&self) -> f32 {
        self.sum() / self.numel() as f32
    }

    /// L2 norm.
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #24: the BF16 KV cache decode must match the F32 KV cache decode
    /// within a BF16-aware tolerance. Same K/V written into both caches, same
    /// query — only the KV *storage* precision differs (attention accumulates
    /// in F32 in both). This is the correctness gate for the BF16 KV path.
    #[test]
    fn bf16_kv_decode_matches_f32_within_tol() {
        if !cuda_available() { eprintln!("SKIP - no CUDA"); return; }
        let (max_seq, n_kv_heads, head_dim, n_heads) = (48usize, 2usize, 16usize, 4usize);
        let kv_dim = n_kv_heads * head_dim;
        let pos = 40usize;

        let mut f32c  = GpuKvCache::new(max_seq, n_kv_heads, head_dim).expect("f32 cache");
        let mut bf16c = GpuKvCache::new_bf16(max_seq, n_kv_heads, head_dim).expect("bf16 cache");
        assert!(!f32c.is_bf16(), "new() must be F32");
        assert!(bf16c.is_bf16(), "new_bf16() must be BF16");
        assert!(bf16c.is_on_gpu(), "bf16 cache must be resident");

        for t in 0..=pos {
            let k: Vec<f32> = (0..kv_dim).map(|i| (((t*7 + i*3) % 29) as f32) * 0.03 - 0.4).collect();
            let v: Vec<f32> = (0..kv_dim).map(|i| (((t*5 + i*2) % 23) as f32) * 0.05 - 0.5).collect();
            let kg = GpuVec::from_slice(&k);
            let vg = GpuVec::from_slice(&v);
            f32c.write(t, &kg, &vg);
            bf16c.write(t, &kg, &vg);
        }

        let q: Vec<f32> = (0..n_heads*head_dim).map(|i| (((i*11) % 17) as f32) * 0.04 - 0.3).collect();
        let qg = GpuVec::from_slice(&q);
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let out_f32  = f32c.decode_attention(&qg, n_heads, pos, scale, 0)
            .expect("f32 decode").download();
        let out_bf16 = bf16c.decode_attention(&qg, n_heads, pos, scale, 0)
            .expect("bf16 decode").download();
        assert_eq!(out_f32.len(), n_heads*head_dim);

        let mut max_abs = 0.0f32;
        for i in 0..out_f32.len() {
            let d = (out_f32[i] - out_bf16[i]).abs();
            max_abs = max_abs.max(d);
            assert!(d <= 2e-2 * out_f32[i].abs().max(1.0) + 5e-3,
                "bf16 KV decode drift at {i}: {} vs {} (|d|={})", out_f32[i], out_bf16[i], d);
        }
        eprintln!("bf16_kv parity: max_abs_diff = {max_abs:.6}");

        // download_keys up-converts BF16 -> F32 and returns the full cache.
        let keys = bf16c.download_keys();
        assert_eq!(keys.len(), max_seq*kv_dim);
    }

    /// #22 Step 2: W4 batched sgemm (dequant-to-scratch + GEMM) must match
    /// the per-column W4 GEMV kernel. The GEMV path is HF-parity verified,
    /// so agreement here is the correctness gate for the new dequant kernel
    /// and the scratch plumbing. Tolerance covers cuBLAS TF32 vs f32 GEMV
    /// low-bit drift (same contract as the f32 batch path).
    #[test]
    fn w4_sgemm_batch_matches_gemv_columns() {
        if !cuda_available() { eprintln!("SKIP - no CUDA"); return; }
        let (m, k, g, n) = (32usize, 64usize, 32usize, 5usize);
        // Deterministic synthetic W4: nibbles cycle through [-8, 6],
        // per-group BF16 scales vary per row/group.
        let words = k / 8;
        let packed: Vec<u32> = (0..m * words).map(|i| {
            let mut w = 0u32;
            for j in 0..8 {
                let nib = ((i * 7 + j * 3) % 15) as u32; // stored nibble 0..14
                w |= nib << (4 * j);
            }
            w
        }).collect();
        let scales: Vec<u16> = (0..m * (k / g)).map(|i| {
            let s = 0.01f32 + 0.003f32 * (i as f32);
            (s.to_bits() >> 16) as u16 // f32 → BF16 bits (truncate)
        }).collect();
        let mat = GpuMatrix::upload_w4(&packed, &scales, m, k, g);
        assert!(mat.is_w4(), "test requires the W4 GPU path");
        // Activations in sgemm layout: [k × n], column j = token j.
        let x: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let mut y_batch = vec![0.0f32; m * n];
        assert!(mat.sgemm(&x, k, n, &mut y_batch),
            "W4 batch sgemm must succeed on GPU (no fallback)");
        for col in 0..n {
            let xcol: Vec<f32> = (0..k).map(|r| x[r * n + col]).collect();
            let mut ycol = vec![0.0f32; m];
            assert!(mat.sgemm(&xcol, k, 1, &mut ycol), "W4 GEMV reference failed");
            for row in 0..m {
                let (a, b) = (y_batch[row * n + col], ycol[row]);
                assert!((a - b).abs() <= 1e-3 * b.abs().max(1.0),
                    "W4 batch/GEMV mismatch at ({row},{col}): {a} vs {b}");
            }
        }
    }

    /// dup() must be an exact copy and must not alias the source buffer.
    /// Runs by default; exercises the device path when CUDA is present and
    /// the host path otherwise.
    #[test]
    fn gpuvec_dup_is_deep_copy() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5 - 250.0).collect();
        let mut a = GpuVec::from_slice(&data);
        let b = a.dup();
        assert_eq!(b.len, a.len);
        assert_eq!(a.is_on_gpu(), b.is_on_gpu(), "dup must stay on the same device");
        assert_eq!(b.download(), data, "dup contents must match source");

        // Mutating the original must not affect the copy (no aliasing).
        let ones = GpuVec::from_slice(&vec![1.0f32; 1000]);
        a.add_inplace(&ones);
        assert_eq!(b.download(), data, "dup must not alias the source buffer");
    }

    #[test]
    fn zeros_shape() {
        let t = Tensor::zeros(&[3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.numel(), 12);
    }

    #[test]
    fn matmul_correct() {
        // [1,2,3;4,5,6] × [7,8;9,10;11,12] = [58,64;139,154]
        let a = Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.], vec![2,3]).unwrap();
        let b = Tensor::from_vec(vec![7.,8.,9.,10.,11.,12.], vec![3,2]).unwrap();
        let c = a.matmul(&b).unwrap();
        let s = c.as_slice().unwrap();
        assert!((s[0] - 58.).abs() < 1e-4);
        assert!((s[3] - 154.).abs() < 1e-4);
    }

    #[test]
    fn softmax_sums_to_one() {
        let t = Tensor::from_vec(vec![1.,2.,3.], vec![1,3]).unwrap();
        let s = t.softmax().unwrap();
        let sum: f32 = s.as_slice().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn transpose_correct() {
        let t = Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.], vec![2,3]).unwrap();
        let tt = t.transpose().unwrap();
        assert_eq!(tt.shape(), &[3,2]);
        assert_eq!(tt.as_slice().unwrap()[1], 4.0); // (0,1)→(1,0) → index 1 in [3,2]
    }

    #[test]
    fn cuda_info() {
        // Just check it doesn't panic; result depends on build environment
        let _ = cuda_available();
        let _ = cuda_device_count();
    }

    #[test]
    fn scale_relu() {
        let t = Tensor::from_vec(vec![-2., -1., 0., 1., 2.], vec![5]).unwrap();
        let s = t.scale(2.0);
        assert_eq!(s.as_slice().unwrap(), &[-4.,-2.,0.,2.,4.]);
        let r = t.relu();
        assert_eq!(r.as_slice().unwrap(), &[0.,0.,0.,1.,2.]);
    }

    #[test]
    fn gpuvec_zeros_len() {
        let v = GpuVec::zeros(64);
        assert_eq!(v.len, 64);
        let data = v.download();
        assert_eq!(data.len(), 64);
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn gpuvec_from_slice_roundtrip() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let v = GpuVec::from_slice(&data);
        assert_eq!(v.len, 16);
        let back = v.download();
        for (a, b) in data.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn gpuvec_add_inplace() {
        let mut a = GpuVec::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b     = GpuVec::from_slice(&[0.5, 0.5, 0.5, 0.5]);
        a.add_inplace(&b);
        let result = a.download();
        assert!((result[0] - 1.5).abs() < 1e-4);
        assert!((result[3] - 4.5).abs() < 1e-4);
    }

    #[test]
    fn gpuvec_rmsnorm_inplace() {
        // RMSNorm of [1,2,3,4] with all-ones weights
        let mut x = GpuVec::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let w = GpuVec::from_slice(&[1.0, 1.0, 1.0, 1.0]);
        x.rmsnorm_inplace(&w, 1e-5);
        let out = x.download();
        // mean(x^2) = (1+4+9+16)/4 = 7.5, rms = sqrt(7.5)
        let rms = (7.5_f32).sqrt();
        assert!((out[0] - 1.0/rms).abs() < 1e-3, "got {}", out[0]);
        assert!((out[3] - 4.0/rms).abs() < 1e-3, "got {}", out[3]);
    }

    #[test]
    fn silu_mul_gpu_correctness() {
        let gate = GpuVec::from_slice(&[0.0, 1.0, -1.0, 2.0]);
        let up   = GpuVec::from_slice(&[1.0, 1.0,  1.0, 1.0]);
        let out = silu_mul_gpu(&gate, &up);
        let result = out.download();
        // silu(0) * 1 = 0 * sigmoid(0) = 0
        assert!((result[0] - 0.0).abs() < 1e-3, "silu(0)*1 = {}", result[0]);
        // silu(1) * 1 = 1*sigmoid(1) ≈ 0.731
        assert!((result[1] - 0.7310586).abs() < 1e-3, "silu(1)*1 = {}", result[1]);
    }

    #[test]
    fn rope_apply_gpu_invertible() {
        // RoPE applied at pos=0 should return the same vector (cos(0)=1, sin(0)=0)
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut v = GpuVec::from_slice(&data);
        rope_apply_gpu(&mut v, 0, 4, 10_000.0);
        let result = v.download();
        for (a, b) in data.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-4, "rope at pos=0 changed value: {} -> {}", a, b);
        }
    }

    #[test]
    fn sgemm_vec_shape() {
        // GpuMatrix: 4×3 (4 out, 3 in), x: 3×1 → out: 4×1
        let w = vec![1.0f32; 4 * 3];
        let gm = GpuMatrix::upload(&w, 4, 3);
        let x = GpuVec::from_slice(&[1.0, 1.0, 1.0]);
        if let Some(out) = gm.sgemm_vec(&x, 1) {
            assert_eq!(out.len, 4);
            let data = out.download();
            // Each output = sum of row = 3.0
            for v in &data { assert!((v - 3.0).abs() < 1e-3, "got {}", v); }
        }
        // If GPU not available, sgemm_vec returns None — that's fine (tested above with CPU fallback)
    }
}
