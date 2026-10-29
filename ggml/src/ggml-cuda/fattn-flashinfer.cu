#ifdef GGML_CUDA_FLASHINFER
#include "fattn-flashinfer.cuh"
#include "convert.cuh"

#define CUTE_DSL_CUDA_ERROR_CHECK(err) CUDA_CHECK(err)
#include "ggml_flashinfer_fp8.h"

struct ggml_cuda_flashinfer_state {
    ggml_flashinfer_fp8_Kernel_Module_t module = {};
    bool loaded[GGML_CUDA_MAX_DEVICES] = {};

    ggml_cuda_flashinfer_state() {
        cudaLibrary_t * library = &module.module;
        cudaError_t status = cudaSuccess;
        struct {
            cudaLibrary_t ** library;
            cudaError_t * status;
        } init = { &library, &status };
        _mlir_ggml_flashinfer_fp8_cuda_init((void **) &init);
        CUDA_CHECK(status);
    }

    void load(int device) {
        if (loaded[device]) {
            return;
        }
        cudaLibrary_t * library = &module.module;
        cudaError_t status = cudaSuccess;
        struct {
            cudaLibrary_t ** library;
            int32_t * device;
            cudaError_t * status;
        } load = { &library, &device, &status };
        _mlir_ggml_flashinfer_fp8_cuda_load_to_device((void **) &load);
        CUDA_CHECK(status);
        loaded[device] = true;
        GGML_LOG_INFO("ggml_cuda_flashinfer: loaded FP8 prefill kernel for CUDA%d\n", device);
    }

    ~ggml_cuda_flashinfer_state() {
        CUDA_CHECK(cudaLibraryUnload(module.module));
    }
};

static std::shared_ptr<ggml_cuda_flashinfer_state> flashinfer_state(int device) {
    static std::mutex mutex;
    static std::weak_ptr<ggml_cuda_flashinfer_state> shared;
    std::lock_guard<std::mutex> lock(mutex);
    auto state = shared.lock();
    if (!state) {
        state = std::make_shared<ggml_cuda_flashinfer_state>();
        shared = state;
    }
    state->load(device);
    return state;
}

bool ggml_cuda_flashinfer_candidate(int device, const ggml_tensor * dst) {
    static const bool disabled = std::getenv("GGML_CUDA_DISABLE_FLASHINFER") != nullptr;
    if (disabled || ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_BLACKWELL) {
        return false;
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (dst->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 ||
        k->type != GGML_TYPE_F8_E4M3 || v->type != GGML_TYPE_F8_E4M3 ||
        q->ne[0] != 256 || k->ne[0] != 256 || v->ne[0] != 256 ||
        q->ne[2] != 24 || k->ne[2] != 4 || v->ne[2] != 4 ||
        q->ne[1] < 128 || q->ne[1] > INT32_MAX/(24*256) || k->ne[1] > INT32_MAX ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        dst->ne[0] != 256 || dst->ne[1] != 24 || dst->ne[2] != q->ne[1] || dst->ne[3] != 1 ||
        q->nb[0] != sizeof(float) || k->ne[1] != v->ne[1] || !ggml_is_contiguous(dst)) {
        return false;
    }
    for (const ggml_tensor * t : {k, v}) {
        if (t->nb[0] != 1 || t->nb[2] != 256 || t->nb[1] != 4*256 || (uintptr_t) t->data % 16 != 0) {
            return false;
        }
    }
    return mask && mask->type == GGML_TYPE_F16 && mask->ne[0] == k->ne[1] &&
        mask->ne[1] >= q->ne[1] && mask->ne[2] == 1 && mask->ne[3] == 1 &&
        dst->src[4] == nullptr && dst->src[5] == nullptr && ggml_get_op_params_f32(dst, 1) == 0.0f &&
        ggml_get_op_params_f32(dst, 2) == 0.0f && ggml_get_op_params_i32(dst, 4) == 0;
}

size_t ggml_cuda_flashinfer_get_alloc_size(const ggml_tensor * dst) {
    const size_t n = ggml_nelements(dst);
    return GGML_PAD(ggml_nbytes(dst), 128) + GGML_PAD(n, 128) + GGML_PAD(n*sizeof(nv_bfloat16), 128) + 4*sizeof(int32_t);
}

static __global__ void flashinfer_pack_q(
        const char * q, uint8_t * packed, int32_t * seq_lens, int n_queries, int n_kv, size_t nb1, size_t nb2) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_queries*24*256) {
        const int d = i % 256;
        const int h = i / 256 % 24;
        const int t = i / (256*24);
        packed[i] = ggml_cuda_fp32_to_f8_e4m3(*(const float *) (q + t*nb1 + h*nb2 + d*sizeof(float)));
    }
    if (i == 0) {
        seq_lens[0] = 0;
        seq_lens[1] = n_queries;
        seq_lens[2] = 0;
        seq_lens[3] = n_kv;
    }
}

bool ggml_cuda_flashinfer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_cuda_flashinfer_candidate(ctx.device, dst)) {
        return false;
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const int n_kv = ggml_get_op_params_i32(dst, 5);
    if (n_kv < q->ne[1] || n_kv > k->ne[1]) {
        return false;
    }
    if (!ctx.flashinfer) {
        // The first uncaptured evaluation loads the library before CUDA graph capture.
        ctx.flashinfer = flashinfer_state(ctx.device);
    }

    const size_t n = ggml_nelements(dst);
    char * scratch = (char *) dst->data + GGML_PAD(ggml_nbytes(dst), 128);
    auto * packed_q = (uint8_t *) scratch;
    auto * output = (nv_bfloat16 *) (scratch + GGML_PAD(n, 128));
    auto * seq_lens = (int32_t *) ((char *) output + GGML_PAD(n*sizeof(nv_bfloat16), 128));
    const int n_queries = q->ne[1];
    flashinfer_pack_q<<<(n + 255)/256, 256, 0, ctx.stream()>>>(
        (const char *) q->data, packed_q, seq_lens, n_queries, n_kv, q->nb[1], q->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    ggml_flashinfer_fp8_Tensor_q_t q_arg = {packed_q, {n_queries}};
    ggml_flashinfer_fp8_Tensor_k_t k_arg = {k->data, {n_kv}};
    ggml_flashinfer_fp8_Tensor_v_t v_arg = {v->data, {n_kv}};
    ggml_flashinfer_fp8_Tensor_o_t o_arg = {output, {n_queries}};
    ggml_flashinfer_fp8_Tensor_cu_q_t cu_q = {seq_lens};
    ggml_flashinfer_fp8_Tensor_cu_k_t cu_k = {seq_lens + 2};
    const float scale_log2 = ggml_get_op_params_f32(dst, 0)*1.4426950408889634f;
    const int status = cute_dsl_ggml_flashinfer_fp8_wrapper(
        &ctx.flashinfer->module, &q_arg, &k_arg, &v_arg, &o_arg, &cu_q, &cu_k, scale_log2, n_queries, ctx.stream());
    GGML_ASSERT(status == 0);
    ggml_get_to_fp32_cuda(GGML_TYPE_BF16)(output, (float *) dst->data, n, ctx.stream());
    return true;
}
#endif
