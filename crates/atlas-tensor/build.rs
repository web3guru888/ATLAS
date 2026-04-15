//! build.rs — Compile CUDA kernels if nvcc is available.
//!
//! This is the zero-dependency GPU strategy:
//!   - `nvcc kernels/matmul.cu` → `libatlas_kernels.a`
//!   - Link against system `libcuda.so` and `libcublas.so`
//!   - Rust calls via `extern "C"` — no cudarc, no tch, no candle
//!
//! If nvcc is not found, CUDA support is silently disabled and
//! atlas-tensor falls back to pure Rust CPU implementation.

use std::process::Command;
use std::path::Path;

fn main() {
    let kernels_dir = Path::new("../../kernels");
    let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| "/tmp".into());

    // Check if nvcc is available
    let nvcc_available = Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !nvcc_available {
        eprintln!("[atlas-tensor/build.rs] nvcc not found — building CPU-only (no CUDA acceleration)");
        println!("cargo:rustc-cfg=feature=\"cpu_only\"");
        return;
    }

    // Compile matmul.cu
    let matmul_cu = kernels_dir.join("matmul.cu");
    let matmul_o  = format!("{}/matmul.o", out_dir);
    if matmul_cu.exists() {
        let status = Command::new("nvcc")
            .args([
                "-O3",
                "-arch=sm_86",           // RTX 3090/4090 target (Ampere)
                "--compiler-options", "-fPIC",
                "-c", matmul_cu.to_str().unwrap(),
                "-o", &matmul_o,
            ])
            .status()
            .expect("nvcc failed");

        if status.success() {
            // Create static library
            Command::new("ar")
                .args(["rcs", &format!("{}/libatlas_kernels.a", out_dir), &matmul_o])
                .status()
                .expect("ar failed");

            println!("cargo:rustc-link-search=native={}", out_dir);
            println!("cargo:rustc-link-lib=static=atlas_kernels");
            println!("cargo:rustc-link-lib=cuda");
            println!("cargo:rustc-cfg=feature=\"cuda\"");
            eprintln!("[atlas-tensor/build.rs] CUDA kernels compiled successfully");
        }
    }

    // Rerun if kernels change
    println!("cargo:rerun-if-changed=../../kernels/matmul.cu");
    println!("cargo:rerun-if-changed=../../kernels/attention.cu");
    println!("cargo:rerun-if-changed=../../kernels/quant.cu");
    println!("cargo:rerun-if-changed=build.rs");
}
