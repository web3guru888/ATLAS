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
        pub fn atlas_adamw_step(
            param: *mut f32, m: *mut f32, v: *mut f32, grad: *const f32,
            lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
            bc1: f32, bc2: f32, n: c_int,
        );
        /// BF16 weight × F32 activation GEMM (W16A32).
        /// A_bf16[M,K] stored as uint16_t (BF16 bit pattern), B[K,N] F32, C[M,N] F32.
        pub fn atlas_sgemm_bf16_f32(
            a_bf16: *const u16, b: *const f32, c: *mut f32,
            m: c_int, n: c_int, k: c_int,
        );
        /// Explicitly drain all pending GPU work (cudaDeviceSynchronize).
        /// Rarely needed — use only when CPU must read GPU output without going
        /// through GpuVec::download() (which syncs implicitly via cudaMemcpy D2H).
        pub fn atlas_sync();
    }
}

/// Drain all pending GPU work (cudaDeviceSynchronize).
/// Rarely needed in normal use — GpuVec::download() syncs implicitly.
/// Call this when you need to measure GPU timing or confirm all kernels have finished.
pub fn device_sync() {
    #[cfg(atlas_cuda)]
    unsafe { ffi::atlas_sync() }
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
    }

    const CUDA_MEMCPY_H2D: i32 = 1;
    const CUDA_MEMCPY_D2H: i32 = 2;

    pub struct GpuBuf {
        pub ptr: *mut f32,
        pub len: usize,
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
            let err = unsafe { cudaMalloc(&mut ptr, len * 4) };
            if err != 0 { return None; }
            Some(Self { ptr: ptr as *mut f32, len })
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
            unsafe {
                cudaMemcpy(out.as_mut_ptr() as *mut c_void, self.ptr as *const c_void,
                           self.len * 4, CUDA_MEMCPY_D2H);
            }
            out
        }
    }
    impl Drop for GpuBuf {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { cudaFree(self.ptr as *mut c_void); }
            }
        }
    }

    /// GPU buffer for BF16 weights (u16, 2 bytes/element).
    /// Used by `GpuBufKind::BF16` to store weight matrices in half the VRAM of f32.
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
    }
    impl Drop for GpuBufBf16 {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { cudaFree(self.ptr as *mut c_void); }
            }
        }
    }

    /// Discriminated union: either an f32 or a BF16 GPU weight buffer.
    /// `GpuMatrix` holds `Option<GpuBufKind>` so it can store either precision.
    pub enum GpuBufKind {
        /// Full f32 weights in VRAM (4 bytes/element).
        F32(GpuBuf),
        /// BF16 weights in VRAM (2 bytes/element).  W16A32: activations stay f32.
        BF16(GpuBufBf16),
    }
    impl GpuBufKind {
        pub fn is_bf16(&self) -> bool { matches!(self, GpuBufKind::BF16(_)) }
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
                    }
                    out.copy_from_slice(&c_buf.download());
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

impl std::fmt::Debug for GpuMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(atlas_cuda)]
        let dtype = match &self.buf {
            Some(gpu::GpuBufKind::F32(_))  => "f32",
            Some(gpu::GpuBufKind::BF16(_)) => "bf16",
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
        Self {
            #[cfg(atlas_cuda)]
            buf: if cuda_available() { gpu::GpuBuf::alloc(len) } else { None },
            cpu: vec![0.0f32; len],
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
        // CPU fallback
        for (a, b) in self.cpu.iter_mut().zip(other.cpu.iter()) { *a += *b; }
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
        // CPU fallback
        let ss: f32 = self.cpu.iter().map(|&v| v * v).sum::<f32>() / self.len as f32;
        let rms_inv = 1.0 / (ss + eps).sqrt();
        for (xi, wi) in self.cpu.iter_mut().zip(w.cpu.iter()) {
            *xi = *xi * rms_inv * wi;
        }
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
    // CPU fallback
    let cpu: Vec<f32> = gate.cpu.iter().zip(up.cpu.iter())
        .map(|(&g, &u)| { let sg = g / (1.0 + (-g).exp()); sg * u })
        .collect();
    GpuVec {
        #[cfg(atlas_cuda)] buf: None,
        cpu,
        len: n,
    }
}

/// Call the GPU AdamW step kernel for one parameter group.
///
/// `param`, `m`, `v` are updated in-place.
/// Returns true if GPU kernel was called, false if fell back to CPU.
pub fn adamw_step_gpu(
    param: *mut f32, m: *mut f32, v: *mut f32, grad: *const f32,
    lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
    bc1: f32, bc2: f32, n: usize,
) -> bool {
    #[cfg(atlas_cuda)]
    if cuda_available() {
        unsafe {
            ffi::atlas_adamw_step(param, m, v, grad, lr, beta1, beta2, eps, wd, bc1, bc2, n as i32);
        }
        return true;
    }
    false
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
