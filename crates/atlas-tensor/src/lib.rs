//! atlas-tensor — The seed of everything.
//!
//! Pure Rust tensor implementation with CUDA FFI for GPU acceleration.
//! No external crates — matmul is either our own CUDA kernel or pure Rust.
//!
//! # Zero-dependency GPU strategy
//!
//! CUDA kernels live in `../../kernels/*.cu`. `build.rs` invokes `nvcc` and
//! links `-lcuda -lcublas` as system libraries. Rust calls them via `extern "C"`.
//!
//! ```text
//! atlas-tensor/build.rs  →  nvcc kernels/matmul.cu  →  libatlas_kernels.a
//!                        →  extern "C" { fn atlas_matmul_f32(...) }
//! ```
//!
//! # First file — the seed of everything
//!
//! Every billion-parameter transformer starts here.

use atlas_core::{AtlasError, Device, DType, Result};

/// A multi-dimensional tensor.
///
/// Data is stored in row-major (C) order on either CPU or GPU.
/// On GPU, `data` is a device pointer (opaque u64) — do not dereference from Rust.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Raw data: on CPU this is actual f32 values; on GPU it is a device pointer.
    data:   Vec<f32>,
    /// Shape of the tensor, e.g. [batch, seq, d_model].
    shape:  Vec<usize>,
    /// Data type.
    dtype:  DType,
    /// Device placement.
    device: Device,
}

impl Tensor {
    /// Create a zero-filled tensor on CPU.
    ///
    /// ```rust
    /// use atlas_tensor::Tensor;
    /// let t = Tensor::zeros(&[2, 3]);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// assert_eq!(t.numel(), 6);
    /// ```
    pub fn zeros(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        Self {
            data:   vec![0.0f32; n],
            shape:  shape.to_vec(),
            dtype:  DType::F32,
            device: Device::Cpu,
        }
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: &[usize], value: f32) -> Self {
        let n = shape.iter().product();
        Self {
            data:   vec![value; n],
            shape:  shape.to_vec(),
            dtype:  DType::F32,
            device: Device::Cpu,
        }
    }

    /// Create a tensor from raw data.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(AtlasError::ShapeMismatch {
                expected: vec![expected],
                got: vec![data.len()],
            });
        }
        Ok(Self { data, shape, dtype: DType::F32, device: Device::Cpu })
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Shape slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// View the raw data (CPU only).
    pub fn as_slice(&self) -> Result<&[f32]> {
        if self.device != Device::Cpu {
            return Err(AtlasError::Other("as_slice called on GPU tensor".into()));
        }
        Ok(&self.data)
    }

    /// Matrix multiplication: [M, K] × [K, N] → [M, N]
    ///
    /// Uses CUDA kernel when CUDA feature is enabled; falls back to pure Rust O(n³).
    /// The CUDA path calls `atlas_matmul_f32` from `kernels/matmul.cu`.
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        // Validate shapes
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(AtlasError::Other("matmul requires 2D tensors".into()));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k != k2 {
            return Err(AtlasError::ShapeMismatch {
                expected: vec![m, k],
                got: vec![k2, n],
            });
        }

        // TODO Stage 1: replace with CUDA kernel call via FFI
        // extern "C" {
        //     fn atlas_matmul_f32(a: *const f32, b: *const f32, c: *mut f32,
        //                         m: i32, n: i32, k: i32);
        // }
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_ip = self.data[i * k + p];
                for j in 0..n {
                    out[i * n + j] += a_ip * other.data[p * n + j];
                }
            }
        }
        Tensor::from_vec(out, vec![m, n])
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(AtlasError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Tensor::from_vec(data, self.shape.clone())
    }

    /// Element-wise multiply.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(AtlasError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
        Tensor::from_vec(data, self.shape.clone())
    }

    /// Scalar multiply.
    pub fn scale(&self, s: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x * s).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype, device: self.device }
    }

    /// Apply ReLU activation.
    pub fn relu(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x.max(0.0)).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype, device: self.device }
    }

    /// Apply softmax along the last dimension.
    pub fn softmax(&self) -> Result<Tensor> {
        if self.ndim() < 1 {
            return Err(AtlasError::Other("softmax requires at least 1D tensor".into()));
        }
        let last = *self.shape.last().unwrap();
        let rows = self.numel() / last;
        let mut data = self.data.clone();
        for r in 0..rows {
            let row = &mut data[r * last..(r + 1) * last];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            row.iter_mut().for_each(|x| *x = (*x - max).exp());
            let sum: f32 = row.iter().sum();
            row.iter_mut().for_each(|x| *x /= sum);
        }
        Tensor::from_vec(data, self.shape.clone())
    }

    /// Reshape (must have same total elements).
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor> {
        let new_n: usize = shape.iter().product();
        if new_n != self.numel() {
            return Err(AtlasError::ShapeMismatch {
                expected: vec![self.numel()],
                got: vec![new_n],
            });
        }
        Ok(Tensor { data: self.data.clone(), shape, dtype: self.dtype, device: self.device })
    }

    /// Transpose a 2D tensor.
    pub fn transpose(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(AtlasError::Other("transpose requires 2D tensor".into()));
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_shape() {
        let t = Tensor::zeros(&[3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.numel(), 12);
        assert!(t.as_slice().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn matmul_2x3_3x2() {
        // [1,2,3; 4,5,6] × [7,8; 9,10; 11,12] = [58,64; 139,154]
        let a = Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.], vec![2,3]).unwrap();
        let b = Tensor::from_vec(vec![7.,8.,9.,10.,11.,12.], vec![3,2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let s = c.as_slice().unwrap();
        assert!((s[0] - 58.0).abs() < 1e-4);
        assert!((s[1] - 64.0).abs() < 1e-4);
        assert!((s[2] - 139.0).abs() < 1e-4);
        assert!((s[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn softmax_sums_to_one() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let s = t.softmax().unwrap();
        let sum: f32 = s.as_slice().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn transpose_2d() {
        let t = Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.], vec![2,3]).unwrap();
        let tt = t.transpose().unwrap();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.as_slice().unwrap()[0], 1.0);
        assert_eq!(tt.as_slice().unwrap()[1], 4.0);
    }

    #[test]
    fn reshape() {
        let t = Tensor::zeros(&[2, 3]);
        let r = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
    }
}
