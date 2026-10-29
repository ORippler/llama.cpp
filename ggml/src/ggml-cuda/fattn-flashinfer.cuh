#pragma once

#ifdef GGML_CUDA_FLASHINFER
#include "common.cuh"

bool ggml_cuda_flashinfer_candidate(int device, const ggml_tensor * dst);
size_t ggml_cuda_flashinfer_get_alloc_size(const ggml_tensor * dst);
bool ggml_cuda_flashinfer(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif
