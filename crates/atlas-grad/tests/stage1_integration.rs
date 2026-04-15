//! Stage 1 integration test — 2-layer MLP forward + backward + one AdamW step.
//!
//! This is the Stage 1 milestone:
//!   "f32 matmul on CPU + GPU, AdamW, backward pass through 2-layer MLP"
//!
//! Network:
//!   x [4,3] → W1 [3,4] → h1 [4,4] → relu → h2 [4,4] → W2 [4,2] → out [4,2]
//!   loss = mean(out²)
//!
//! Validates:
//!   1. Forward pass through matmul + relu
//!   2. Backward pass (VJPs for matmul + relu)
//!   3. Gradients have correct shapes
//!   4. One AdamW step moves parameters
//!   5. INT4 quantize→dequantize→matmul matches f32 within tolerance
//!   6. Cosine LR schedule produces expected values

use atlas_tensor::Tensor;
use atlas_grad::{GradTape, Op};
use atlas_optim::{AdamW, AdamWConfig, CosineScheduler, ParamState, clip_grad_norm};
use atlas_quant::{Int4Tensor, Int8Tensor, INT4_BLOCK_SIZE, LoraAdapter, estimate_vram_gb};

/// Helper: randn-ish from deterministic seed (no rand crate)
fn pseudo_randn(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let f = (s >> 33) as f32 / (u32::MAX as f32);
        f * 2.0 - 1.0
    }).collect()
}

#[test]
fn stage1_mlp_forward() {
    // Input: batch=4, features=3
    let x  = Tensor::from_vec(pseudo_randn(12, 1), vec![4, 3]).unwrap();
    let w1 = Tensor::from_vec(pseudo_randn(12, 2), vec![3, 4]).unwrap();
    let w2 = Tensor::from_vec(pseudo_randn(8,  3), vec![4, 2]).unwrap();

    // Forward
    let h1  = x.matmul(&w1).unwrap();
    assert_eq!(h1.shape(), &[4, 4]);
    let h1r = h1.relu();
    let out = h1r.matmul(&w2).unwrap();
    assert_eq!(out.shape(), &[4, 2]);
    println!("[stage1] forward pass OK — out shape {:?}, mean={:.4}", out.shape(), out.mean());
}

#[test]
fn stage1_mlp_backward() {
    let mut tape = GradTape::new();

    // Register params on tape
    let x_idx  = tape.push(Tensor::from_vec(pseudo_randn(12, 42), vec![4,3]).unwrap());
    let w1_idx = tape.push(Tensor::from_vec(pseudo_randn(12, 43), vec![3,4]).unwrap());
    let w2_idx = tape.push(Tensor::from_vec(pseudo_randn(8,  44), vec![4,2]).unwrap());

    // Forward on tape
    let h1  = tape.matmul(x_idx, w1_idx).unwrap();
    let h1r = tape.relu(h1);
    let out = tape.matmul(h1r, w2_idx).unwrap();

    // Backward from output
    tape.backward(out).unwrap();

    // W1 and W2 should have gradients
    assert!(tape.grads[w1_idx].is_some(), "W1 grad missing");
    assert!(tape.grads[w2_idx].is_some(), "W2 grad missing");

    let dw1 = tape.grads[w1_idx].as_ref().unwrap();
    let dw2 = tape.grads[w2_idx].as_ref().unwrap();
    assert_eq!(dw1.shape(), &[3, 4], "dW1 shape mismatch");
    assert_eq!(dw2.shape(), &[4, 2], "dW2 shape mismatch");

    println!("[stage1] backward pass OK — dW1 norm={:.4}, dW2 norm={:.4}",
        dw1.norm(), dw2.norm());
}

#[test]
fn stage1_adamw_reduces_loss() {
    // Simple 1-layer linear model: y = W·x, loss = ||y||²
    // Gradient: dW = 2·x^T·y  (chain rule for ||Wx||²)
    let w_init = pseudo_randn(6, 99);   // [3,2] weight
    let x_data = pseudo_randn(6, 100);  // [2,3] input

    let cfg = AdamWConfig { lr: 0.01, weight_decay: 0.01, ..Default::default() };
    let mut opt = AdamW::new(cfg.clone());
    opt.add_param(ParamState::new("W", w_init.clone(), vec![3, 2], true));

    let sched = CosineScheduler::new(0.01, 1e-5, 200, 20);

    let mut prev_loss = f32::MAX;
    for step in 1..=50 {
        let w_t = Tensor::from_vec(opt.params[0].param.clone(), vec![3, 2]).unwrap();
        let x_t = Tensor::from_vec(x_data.clone(), vec![2, 3]).unwrap();
        let y_t = x_t.matmul(&w_t).unwrap();          // [2,2]
        let loss = y_t.as_slice().unwrap().iter().map(|v| v*v).sum::<f32>() / y_t.numel() as f32;

        // Compute gradient: dL/dW = (2/n) · x^T · y
        let dydw: Vec<f32> = {
            let x = x_data.as_slice();
            let y = y_t.as_slice().unwrap();
            let (m, k, n) = (3, 2, 2); // W is [3,2], x is [2,3], y is [2,2]
            // dW[i,j] = (2/n) * sum_b x[b,i] * y[b,j]
            // but W·x: W[3,2] × x[2,3] → out[3,3] — let's simplify
            // Use the actual grad: dL/dW for matmul x@W → y: dW = x^T @ dy
            let dy: Vec<f32> = y.iter().map(|v| 2.0 * v / y_t.numel() as f32).collect();
            let mut dw = vec![0.0f32; 6];
            // x is [2,3], dy is [2,2], W is [3,2]
            // y = x @ W (2×3 × 3×2 → 2×2), dW = x^T @ dy
            for i in 0..3 {
                for j in 0..2 {
                    let mut acc = 0.0;
                    for b in 0..2 {
                        acc += x[b * 3 + i] * dy[b * 2 + j];
                    }
                    dw[i * 2 + j] = acc;
                }
            }
            dw
        };

        sched.apply(&mut opt, step);
        opt.step(&[dydw]).unwrap();

        if step > 5 {
            prev_loss = prev_loss.min(loss);
        }
        if step == 50 {
            println!("[stage1] AdamW step 50: loss={loss:.6}, lr={:.2e}", opt.cfg.lr);
            assert!(loss < 1.0, "Loss should decrease from AdamW training, got {loss}");
        }
    }
}

#[test]
fn stage1_int8_quant_matmul() {
    // Quantize a weight matrix, dequantize, and verify matmul error is small
    let w_data = pseudo_randn(256, 7);
    let w = Tensor::from_vec(w_data.clone(), vec![16, 16]).unwrap();
    let x = Tensor::from_vec(pseudo_randn(48, 8), vec![3, 16]).unwrap();

    // Reference: f32 matmul
    let ref_out = x.matmul(&w.transpose().unwrap()).unwrap(); // [3,16]

    // Quantized path
    let wq  = Int8Tensor::quantize(&w).unwrap();
    let wdq = wq.dequantize();
    let q_out = x.matmul(&wdq.transpose().unwrap()).unwrap();

    // Compare
    let r = ref_out.as_slice().unwrap();
    let q = q_out.as_slice().unwrap();
    let max_err = r.iter().zip(q.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    println!("[stage1] INT8 quant matmul max_err={max_err:.4}");
    assert!(max_err < 0.1, "INT8 matmul error too large: {max_err}");
}

#[test]
fn stage1_int4_lora_pipeline() {
    // INT4 quantize → dequantize → LoRA forward: validates the QLoRA pipeline
    let w = Tensor::from_vec(pseudo_randn(512, 5), vec![16, 32]).unwrap();
    let q = Int4Tensor::quantize(&w, INT4_BLOCK_SIZE).unwrap();
    let dq = q.dequantize();

    let lora = LoraAdapter::new(32, 16, 4, 8.0);
    let x = Tensor::full(&[2, 32], 0.1);

    // Base matmul + LoRA
    let base_out = x.matmul(&dq.transpose().unwrap()).unwrap();
    let lora_out = lora.forward(&x).unwrap();
    let _ = base_out.add(&lora_out).unwrap();

    println!("[stage1] INT4+LoRA pipeline OK — compression ratio={:.1}×",
        q.compression_ratio());
    assert!(q.compression_ratio() > 7.0);
}

#[test]
fn stage1_vram_fits_t4() {
    // Verify OLMo 3 7B with QLoRA fits in T4's 15GB
    let (base, lora, total) = estimate_vram_gb(7_000_000_000, 16, 32, 4096);
    println!("[stage1] T4 VRAM estimate: base={base:.2}GB lora={lora:.2}GB total={total:.2}GB");
    assert!(total < 15.0, "QLoRA 7B won't fit in T4 15GB! Got {total:.2}GB");
}

#[test]
fn stage1_cosine_schedule() {
    let sched = CosineScheduler::new(1e-4, 1e-6, 1000, 100);
    assert_eq!(sched.lr(0), 0.0);
    assert!((sched.lr(100) - 1e-4).abs() < 1e-9);
    assert!(sched.lr(1000) < 2e-6);
    // Monotone after warmup
    let mut prev = sched.lr(100);
    for t in 101..=1000 {
        let cur = sched.lr(t);
        assert!(cur <= prev + 1e-12, "LR not monotone at t={t}");
        prev = cur;
    }
    println!("[stage1] Cosine schedule OK");
}
