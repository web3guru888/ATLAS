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
    }
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
}


// ── GpuMatrix — weight matrix pinned in VRAM ──────────────────────────────

/// A matrix pre-uploaded to GPU VRAM (upload once, multiply many times).
///
/// On CUDA builds: the data lives in VRAM and is reused across `sgemm()` calls.
/// Input vectors are uploaded per call; only the tiny x-vector moves on the bus.
///
/// On CPU-only builds (no `atlas_cuda` cfg): this is a zero-overhead no-op;
/// `sgemm()` always returns `false` and the caller uses its own CPU path.
pub struct GpuMatrix {
    #[cfg(atlas_cuda)]
    buf: Option<gpu::GpuBuf>,
    pub rows: usize,   // output dimension
    pub cols: usize,   // input dimension
}

impl GpuMatrix {
    /// Upload a row-major f32 matrix [rows × cols] to GPU VRAM.
    /// Falls back gracefully if CUDA is not available.
    pub fn upload(data: &[f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self {
            #[cfg(atlas_cuda)]
            buf: if cuda_available() { gpu::GpuBuf::upload(data) } else { None },
            rows,
            cols,
        }
    }

    /// Whether the matrix is resident in GPU VRAM.
    pub fn is_on_gpu(&self) -> bool {
        #[cfg(atlas_cuda)]
        { self.buf.is_some() }
        #[cfg(not(atlas_cuda))]
        { false }
    }

    /// GPU SGEMM: `out[m × n] = self[m × k] × rhs[k × n]` (row-major).
    ///
    /// Weight matrix is already in VRAM. Only `rhs` (the input activations)
    /// is uploaded per call — typically a tiny x-vector (k floats).
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
                    unsafe {
                        ffi::atlas_matmul_f32(
                            a_buf.ptr, b_buf.ptr, c_buf.ptr,
                            m as i32, n as i32, k as i32,
                        );
                    }
                    out.copy_from_slice(&c_buf.download());
                    return true;
                }
            }
        }
        false
    }
}

impl std::fmt::Debug for GpuMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuMatrix({}×{}, gpu={})", self.rows, self.cols, self.is_on_gpu())
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
}
