//! build.rs — Compile CUDA kernels if nvcc is available.
//!
//! Zero-dependency GPU strategy:
//!   - `nvcc kernels/matmul.cu` → compiled into the build output
//!   - Link against system `libcuda` and `libcublas` (system libs, not Rust crates)
//!   - Rust calls via `extern "C"` — no cudarc, no tch, no candle
//!
//! GPU architecture auto-detection order:
//!   1. ATLAS_CUDA_ARCH env var (e.g. "sm_75")
//!   2. Query `nvidia-smi` for the first GPU's compute capability
//!   3. Fall back to sm_75 (T4 / Turing — conservative safe default)
//!
//! If nvcc is not found, CUDA support is silently disabled and
//! atlas-tensor falls back to the pure Rust CPU implementation.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Declare custom cfg flags so rustc's check-cfg lint doesn't warn
    println!("cargo::rustc-check-cfg=cfg(atlas_cuda)");
    println!("cargo::rustc-check-cfg=cfg(atlas_cpu_only)");

    // Emit rerun triggers
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ATLAS_CUDA_ARCH");
    println!("cargo:rerun-if-changed=../../kernels/matmul.cu");
    println!("cargo:rerun-if-changed=../../kernels/attention.cu");
    println!("cargo:rerun-if-changed=../../kernels/quant.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kernels_dir = Path::new("../../kernels");
    let matmul_cu = kernels_dir.join("matmul.cu");

    // ── 1. Locate nvcc ────────────────────────────────────────────────────────
    let nvcc = find_nvcc();
    let Some(nvcc) = nvcc else {
        eprintln!("[atlas-tensor/build.rs] nvcc not found — CPU-only build");
        println!("cargo:rustc-cfg=atlas_cpu_only");
        return;
    };
    eprintln!("[atlas-tensor/build.rs] nvcc = {}", nvcc.display());

    // ── 2. Determine GPU architecture ─────────────────────────────────────────
    let arch = gpu_arch();
    eprintln!("[atlas-tensor/build.rs] GPU arch = {arch}");

    // ── 3. Compile matmul.cu ─────────────────────────────────────────────────
    if !matmul_cu.exists() {
        eprintln!("[atlas-tensor/build.rs] kernels/matmul.cu not found — skipping CUDA");
        println!("cargo:rustc-cfg=atlas_cpu_only");
        return;
    }

    let obj = out_dir.join("matmul.o");
    let lib = out_dir.join("libatlas_kernels.a");

    // Compile .cu → .o
    let status = Command::new(&nvcc)
        .args([
            "-O3",
            &format!("-arch={arch}"),
            "--compiler-options", "-fPIC",
            "-c", matmul_cu.to_str().unwrap(),
            "-o", obj.to_str().unwrap(),
        ])
        .status()
        .expect("nvcc invocation failed");

    if !status.success() {
        eprintln!("[atlas-tensor/build.rs] nvcc compilation FAILED — falling back to CPU");
        println!("cargo:rustc-cfg=atlas_cpu_only");
        return;
    }

    // Package .o → static lib
    let ar_ok = Command::new("ar")
        .args(["rcs", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !ar_ok {
        eprintln!("[atlas-tensor/build.rs] ar failed — CPU-only");
        println!("cargo:rustc-cfg=atlas_cpu_only");
        return;
    }

    // Tell cargo where to find it
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // Also add CUDA lib directories so the linker can find libcudart
    for dir in &[
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12.9/lib64",
        "/usr/local/cuda-12.8/lib64",
        "/usr/local/cuda-12.6/lib64",
        "/usr/local/cuda-12.0/lib64",
        "/usr/local/cuda-11.8/lib64",
    ] {
        if std::path::Path::new(dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
    println!("cargo:rustc-link-lib=static=atlas_kernels");
    println!("cargo:rustc-link-lib=cudart");  // CUDA runtime (malloc/free/memcpy/sync/kernels)
    println!("cargo:rustc-link-lib=cublas");  // cuBLAS — tensor core GEMM (system lib, not a Rust crate)
    println!("cargo:rustc-cfg=atlas_cuda");
    eprintln!("[atlas-tensor/build.rs] CUDA kernels compiled OK ({arch})");
}

/// Find nvcc: check ATLAS_CUDA_ARCH env hint path, common CUDA install dirs, then PATH.
fn find_nvcc() -> Option<PathBuf> {
    // Common install locations
    let candidates = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.9/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12.4/bin/nvcc",
        "/usr/local/cuda-12.2/bin/nvcc",
        "/usr/local/cuda-12.0/bin/nvcc",
        "/usr/local/cuda-11.8/bin/nvcc",
        "nvcc", // already in PATH
    ];
    for c in &candidates {
        let path = PathBuf::from(c);
        let found = if path.is_absolute() {
            path.exists()
        } else {
            Command::new(c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
        };
        if found {
            return Some(path);
        }
    }
    None
}

/// Determine the best CUDA architecture string.
/// Priority: ATLAS_CUDA_ARCH env → nvidia-smi query → safe default (sm_75).
fn gpu_arch() -> String {
    // 1. Explicit env override
    if let Ok(arch) = std::env::var("ATLAS_CUDA_ARCH") {
        if !arch.is_empty() {
            return arch;
        }
    }

    // 2. Ask nvidia-smi for compute capability of GPU 0
    //    Output format: "7.5" for T4, "8.6" for RTX 3090/4090, etc.
    let smi = Command::new("nvidia-smi")
        .args([
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
            "--id=0",
        ])
        .output();

    if let Ok(out) = smi {
        if out.status.success() {
            let raw = String::from_utf8_lossy(&out.stdout);
            let cc = raw.trim().replace('.', "");
            if !cc.is_empty() && cc.chars().all(|c| c.is_ascii_digit()) {
                return format!("sm_{cc}");
            }
        }
    }

    // 3. Safe fallback — T4 (Turing, sm_75).
    //    Works on T4, also compiles (with reduced perf) on newer cards.
    "sm_75".to_string()
}
