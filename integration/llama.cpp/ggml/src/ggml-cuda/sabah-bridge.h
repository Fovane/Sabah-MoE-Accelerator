#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_cuda_sabah_mul_mat_id(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * ids,
    ggml_tensor * dst,
    void * stream);


void ggml_cuda_sabah_compare_after_fallback(
    const ggml_tensor * dst,
    void * stream);
#ifdef __cplusplus
}
#endif