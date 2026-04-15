//! atlas-grad — Reverse-mode automatic differentiation tape.
//!
//! Records forward-pass operations on a `GradTape`. Calling `.backward()`
//! traverses the tape in reverse and accumulates gradients into each tensor.
//!
//! # Stage 1 milestone
//! Working backward pass through a 2-layer MLP. That validates the entire
//! gradient pipeline before the transformer is built.
//!
//! # Design (no external deps)
//! The tape is a `Vec<Op>` where each `Op` stores:
//!   - the operation type (matmul, add, relu, ...)
//!   - indices into a flat arena of tensors
//!   - a closure computing the VJP (vector-Jacobian product)

use atlas_core::Result;
use atlas_tensor::Tensor;

/// A recorded operation on the tape.
#[derive(Debug)]
pub enum Op {
    /// Matrix multiply: output = left × right
    Matmul { left: usize, right: usize, out: usize },
    /// Element-wise add: out = a + b
    Add    { a: usize, b: usize, out: usize },
    /// ReLU: out = relu(inp)
    Relu   { inp: usize, out: usize },
    /// Softmax: out = softmax(inp)
    Softmax { inp: usize, out: usize },
}

/// The gradient tape — records forward computation for later backward pass.
#[derive(Default)]
pub struct GradTape {
    /// Flat arena of all tensors (inputs, intermediates, outputs).
    pub tensors: Vec<Tensor>,
    /// Gradients parallel to `tensors`.
    pub grads:   Vec<Option<Tensor>>,
    /// Recorded operations in forward order.
    pub ops:     Vec<Op>,
}

impl GradTape {
    /// Create a new empty tape.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tensor on the tape, returning its index.
    pub fn push(&mut self, t: Tensor) -> usize {
        let idx = self.tensors.len();
        self.grads.push(None);
        self.tensors.push(t);
        idx
    }

    /// Record a matmul operation and return the output tensor index.
    pub fn matmul(&mut self, left: usize, right: usize) -> Result<usize> {
        let out_t = self.tensors[left].matmul(&self.tensors[right])?;
        let out = self.push(out_t);
        self.ops.push(Op::Matmul { left, right, out });
        Ok(out)
    }

    /// Record a ReLU operation and return the output tensor index.
    pub fn relu(&mut self, inp: usize) -> usize {
        let out_t = self.tensors[inp].relu();
        let out = self.push(out_t);
        self.ops.push(Op::Relu { inp, out });
        out
    }

    /// Run backward pass from `loss_idx`, seeding with all-ones gradient.
    ///
    /// This is the standard backward pass for a scalar loss.
    pub fn backward(&mut self, loss_idx: usize) -> Result<()> {
        let loss_shape = self.tensors[loss_idx].shape().to_vec();
        let seed = Tensor::full(&loss_shape, 1.0);
        self.backward_with_grad(loss_idx, seed)
    }

    /// Run backward pass from `idx` with a caller-supplied initial gradient.
    ///
    /// Use this when the loss is computed externally (e.g., cross-entropy on
    /// logits) and you already know the gradient at this point in the graph.
    ///
    /// # Example
    /// ```rust
    /// use atlas_grad::GradTape;
    /// use atlas_tensor::Tensor;
    /// let mut tape = GradTape::new();
    /// let x = tape.push(Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap());
    /// let y = tape.relu(x);
    /// let custom = Tensor::from_vec(vec![0.5, 3.0], vec![1, 2]).unwrap();
    /// tape.backward_with_grad(y, custom).unwrap();
    /// ```
    pub fn backward_with_grad(&mut self, idx: usize, grad: Tensor) -> Result<()> {
        self.grads[idx] = Some(grad);
        for op in self.ops.iter().rev() {
            match op {
                Op::Matmul { left, right, out } => {
                    // d_left  = d_out × right^T
                    // d_right = left^T × d_out
                    if let Some(d_out) = &self.grads[*out] {
                        let rt = self.tensors[*right].transpose()?;
                        let lt = self.tensors[*left].transpose()?;
                        let d_left  = d_out.matmul(&rt)?;
                        let d_right = lt.matmul(d_out)?;
                        self.grads[*left]  = Some(d_left);
                        self.grads[*right] = Some(d_right);
                    }
                }
                Op::Relu { inp, out } => {
                    // d_inp = d_out * (inp > 0)
                    if let Some(d_out) = &self.grads[*out] {
                        let mask: Vec<f32> = self.tensors[*inp]
                            .as_slice()?
                            .iter()
                            .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
                            .collect();
                        let mask_t = Tensor::from_vec(mask, self.tensors[*inp].shape().to_vec())?;
                        self.grads[*inp] = Some(d_out.mul(&mask_t)?);
                    }
                }
                _ => { /* TODO: implement remaining VJPs */ }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_tensor::Tensor;

    #[test]
    fn forward_relu() {
        let mut tape = GradTape::new();
        let x = tape.push(Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![1, 4]).unwrap());
        let y = tape.relu(x);
        let s = tape.tensors[y].as_slice().unwrap();
        assert_eq!(s, &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn backward_relu_mask() {
        let mut tape = GradTape::new();
        let x = tape.push(Tensor::from_vec(vec![-1.0, 2.0], vec![1, 2]).unwrap());
        let y = tape.relu(x);
        tape.backward(y).unwrap();
        let grad = tape.grads[x].as_ref().unwrap().as_slice().unwrap().to_vec();
        // gradient through relu: 0 for -1, 1 for 2
        assert_eq!(grad, vec![0.0, 1.0]);
    }

    #[test]
    fn backward_with_custom_grad() {
        let mut tape = GradTape::new();
        let x = tape.push(Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let y = tape.relu(x);
        // Custom gradient instead of all-ones
        let custom = Tensor::from_vec(vec![0.5, 3.0], vec![1, 2]).unwrap();
        tape.backward_with_grad(y, custom).unwrap();
        let grad = tape.grads[x].as_ref().unwrap().as_slice().unwrap().to_vec();
        // Both inputs positive → mask = [1,1], gradient = custom * mask
        assert_eq!(grad, vec![0.5, 3.0]);
    }

    #[test]
    fn backward_with_grad_through_matmul() {
        let mut tape = GradTape::new();
        // x=[1,2], W=[[1,0],[0,1]] (identity) → y = x×W = [1,2]
        let x = tape.push(Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let w = tape.push(Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap());
        let y = tape.matmul(x, w).unwrap();
        let d_y = Tensor::from_vec(vec![0.5, -0.5], vec![1, 2]).unwrap();
        tape.backward_with_grad(y, d_y).unwrap();
        // d_x = d_y × W^T = [0.5, -0.5] × I = [0.5, -0.5]
        let dx = tape.grads[x].as_ref().unwrap().as_slice().unwrap();
        assert!((dx[0] - 0.5).abs() < 1e-6);
        assert!((dx[1] - (-0.5)).abs() < 1e-6);
        // d_W = x^T × d_y = [[1],[2]] × [[0.5,-0.5]] = [[0.5,-0.5],[1.0,-1.0]]
        let dw = tape.grads[w].as_ref().unwrap().as_slice().unwrap();
        assert!((dw[0] - 0.5).abs() < 1e-6);
        assert!((dw[1] - (-0.5)).abs() < 1e-6);
        assert!((dw[2] - 1.0).abs() < 1e-6);
        assert!((dw[3] - (-1.0)).abs() < 1e-6);
    }
}
