/*
 * kernels/quant.cu — INT4/INT8 quantization kernels (Stage 1)
 *
 * Implements:
 *   - atlas_quantize_int8_f32:  f32 → INT8 with per-tensor scale
 *   - atlas_dequantize_int8_f32: INT8 → f32
 *   - atlas_quantize_int4_f32:  f32 → INT4 packed (2 values per byte)
 *   - atlas_dequantize_int4_f32: INT4 → f32
 *
 * Used by atlas-quant for QLoRA (4-bit base model + f16 LoRA adapters).
 * Enables 7B model training on 24GB VRAM (RTX 4090).
 *
 * TODO Stage 1: implement quantization kernels from bitsandbytes source.
 */

#include <cuda_runtime.h>
#include <stdint.h>

/* INT8 quantize: scale = max(|x|) / 127 */
extern "C" void atlas_quantize_int8(
    const float*   __restrict__ input,
    int8_t*        __restrict__ output,
    float*         __restrict__ scale,  /* output: one scale per row */
    int n_rows, int n_cols
) {
    /* TODO Stage 1 */
    (void)input; (void)output; (void)scale;
    (void)n_rows; (void)n_cols;
}

/* INT8 dequantize: x_f32 = x_int8 * scale */
extern "C" void atlas_dequantize_int8(
    const int8_t*  __restrict__ input,
    const float*   __restrict__ scale,
    float*         __restrict__ output,
    int n_rows, int n_cols
) {
    /* TODO Stage 1 */
    (void)input; (void)scale; (void)output;
    (void)n_rows; (void)n_cols;
}

/* INT4 quantize: packs two 4-bit values per byte */
extern "C" void atlas_quantize_int4(
    const float*   __restrict__ input,
    uint8_t*       __restrict__ output, /* packed: n/2 bytes */
    float*         __restrict__ scale,
    int n
) {
    /* TODO Stage 1 */
    (void)input; (void)output; (void)scale; (void)n;
}
