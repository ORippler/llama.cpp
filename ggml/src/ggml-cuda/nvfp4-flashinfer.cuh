#pragma once

#include "common.cuh"

static inline bool ggml_cuda_flashinfer_arch_supported(int cc) {
    return cc == GGML_CUDA_CC_BLACKWELL || cc == GGML_CUDA_CC_GB10;
}

static __host__ __device__ inline size_t ggml_cuda_nvfp4_scale_index(int64_t row, int64_t col, int64_t k, int64_t n) {
    GGML_UNUSED(n);
    return ((row / 128) * (k / 64) + col / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + col % 4;
}

ggml_cuda_nvfp4_layout ggml_cuda_nvfp4_get_layout(const ggml_tensor * tensor);
const ggml_tensor * ggml_cuda_nvfp4_get_aux(const ggml_tensor * tensor, const char * suffix);
bool ggml_cuda_nvfp4_repack_enabled(int device);
bool ggml_cuda_nvfp4_repack_candidate(const ggml_tensor * tensor);
bool ggml_cuda_nvfp4_q4_allowed(const ggml_tensor * node);
void ggml_cuda_nvfp4_set(const ggml_cuda_nvfp4_layout & layout, const void * src, size_t offset, size_t size, cudaStream_t stream);
void ggml_cuda_nvfp4_get(const ggml_cuda_nvfp4_layout & layout, void * dst, size_t offset, size_t size, cudaStream_t stream);
void ggml_cuda_nvfp4_unpack(const ggml_cuda_nvfp4_layout & layout, void * dst, size_t offset, size_t size, cudaStream_t stream);
void ggml_cuda_nvfp4_pack(const ggml_cuda_nvfp4_layout & layout, const void * src, size_t offset, size_t size, cudaStream_t stream);
bool ggml_cuda_flashinfer_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * dst);
bool ggml_cuda_flashinfer_mul_mat_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * matmul, const ggml_tensor * scale, ggml_tensor * dst);
bool ggml_cuda_flashinfer_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * up_scale, const ggml_tensor * gate_scale, ggml_tensor * dst);

struct ggml_cuda_flashinfer_moe_args {
    const ggml_tensor * up = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * down = nullptr;
    const ggml_tensor * input = nullptr;
    const ggml_tensor * ids = nullptr;
    const ggml_tensor * weights = nullptr;
    ggml_tensor * dst = nullptr;
};

bool ggml_cuda_flashinfer_moe(ggml_backend_cuda_context & ctx, const ggml_cuda_flashinfer_moe_args & args);
bool ggml_cuda_nvfp4_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * weight, const ggml_tensor * input,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);

struct ggml_cuda_nvfp4_unpacked {
    ggml_cuda_pool_alloc<char> allocation;
    ggml_tensor tensor;

    ggml_cuda_nvfp4_unpacked(ggml_backend_cuda_context & ctx, const ggml_tensor * src);
    const ggml_tensor * get() const { return &tensor; }
};
