#ifdef GGML_CUDA_FLASHINFER
#include "fattn-flashinfer.cuh"
#include "convert.cuh"

#include <mutex>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

struct ggml_cuda_flashinfer_state {
    tvm::ffi::Optional<tvm::ffi::Module> module;
    tvm::ffi::Function run;
    tvm::ffi::Function run_16_2;

    ggml_cuda_flashinfer_state() {
        const char * path = std::getenv("GGML_CUDA_FLASHINFER_LIBRARY");
        try {
            module = tvm::ffi::Module::LoadFromFile(path ? path : GGML_CUDA_FLASHINFER_LIBRARY);
            auto function = module.value()->GetFunction("ggml_flashinfer_fp8_v1");
            if (function.has_value()) {
                run = function.value();
            }
            auto moe_attention = module.value()->GetFunction("ggml_flashinfer_fp8_h16_kv2_v1");
            if (moe_attention.has_value()) {
                run_16_2 = moe_attention.value();
            }
            GGML_LOG_INFO("ggml_cuda_flashinfer: loaded TVM FFI library (attention_24_4=%d, attention_16_2=%d)\n", run.defined(), run_16_2.defined());
        } catch (const std::exception & e) {
            GGML_LOG_WARN("ggml_cuda_flashinfer: TVM FFI unavailable: %s\n", e.what());
        }
    }
};

static std::shared_ptr<ggml_cuda_flashinfer_state> flashinfer_state() {
    static std::mutex mutex;
    static std::weak_ptr<ggml_cuda_flashinfer_state> shared;
    std::lock_guard<std::mutex> lock(mutex);
    auto state = shared.lock();
    if (!state) {
        state = std::make_shared<ggml_cuda_flashinfer_state>();
        shared = state;
    }
    return state;
}

struct flashinfer_stream_guard {
    int device;
    TVMFFIStreamHandle previous;

    flashinfer_stream_guard(int device, cudaStream_t stream) : device(device) {
        GGML_ASSERT(TVMFFIEnvSetStream(kDLCUDA, device, stream, &previous) == 0);
    }

    ~flashinfer_stream_guard() {
        GGML_ASSERT(TVMFFIEnvSetStream(kDLCUDA, device, previous, nullptr) == 0);
    }
};

bool ggml_cuda_flashinfer_candidate(int device, const ggml_tensor * dst) {
    static const bool disabled = std::getenv("GGML_CUDA_DISABLE_FLASHINFER") != nullptr;
    const int cc = ggml_cuda_info().devices[device].cc;
    if (disabled || (cc != GGML_CUDA_CC_BLACKWELL && cc != GGML_CUDA_CC_GB10)) {
        return false;
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (dst->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 ||
        k->type != GGML_TYPE_F8_E4M3 || v->type != GGML_TYPE_F8_E4M3 ||
        q->ne[0] != 256 || k->ne[0] != 256 || v->ne[0] != 256 ||
        ((q->ne[2] != 24 || k->ne[2] != 4) && (q->ne[2] != 16 || k->ne[2] != 2)) ||
        v->ne[2] != k->ne[2] ||
        q->ne[1] < 128 || q->ne[1] > INT32_MAX/(q->ne[2]*256) || k->ne[1] > INT32_MAX ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        dst->ne[0] != 256 || dst->ne[1] != q->ne[2] || dst->ne[2] != q->ne[1] || dst->ne[3] != 1 ||
        q->nb[0] != sizeof(float) || k->ne[1] != v->ne[1] || !ggml_is_contiguous(dst)) {
        return false;
    }
    for (const ggml_tensor * t : {k, v}) {
        if (t->nb[0] != 1 || t->nb[2] != 256 || t->nb[1] != size_t(k->ne[2])*256 || (uintptr_t) t->data % 16 != 0) {
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
        const char * q, uint8_t * packed, int32_t * seq_lens, int n_queries, int n_kv, int n_heads, size_t nb1, size_t nb2) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_queries*n_heads*256) {
        const int d = i % 256;
        const int h = i / 256 % n_heads;
        const int t = i / (256*n_heads);
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
        ctx.flashinfer = flashinfer_state();
    }

    const auto & run = q->ne[2] == 24 ? ctx.flashinfer->run : ctx.flashinfer->run_16_2;
    if (!run.defined()) {
        return false;
    }

    const size_t n = ggml_nelements(dst);
    char * scratch = (char *) dst->data + GGML_PAD(ggml_nbytes(dst), 128);
    auto * packed_q = (uint8_t *) scratch;
    auto * output = (nv_bfloat16 *) (scratch + GGML_PAD(n, 128));
    auto * seq_lens = (int32_t *) ((char *) output + GGML_PAD(n*sizeof(nv_bfloat16), 128));
    const int n_queries = q->ne[1];
    flashinfer_pack_q<<<(n + 255)/256, 256, 0, ctx.stream()>>>(
        (const char *) q->data, packed_q, seq_lens, n_queries, n_kv, q->ne[2], q->nb[1], q->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    int64_t q_shape[] = {n_queries, q->ne[2], 256};
    int64_t kv_shape[] = {n_kv, k->ne[2], 256};
    int64_t q_strides[] = {q->ne[2]*256, 256, 1};
    int64_t kv_strides[] = {k->ne[2]*256, 256, 1};
    int64_t cu_shape[] = {2};
    int64_t cu_strides[] = {1};
    const DLDevice device = {kDLCUDA, ctx.device};
    const DLDataType fp8 = {kDLFloat8_e4m3fn, 8, 1};
    DLTensor q_arg = {packed_q, device, 3, fp8, q_shape, q_strides, 0};
    DLTensor k_arg = {k->data, device, 3, fp8, kv_shape, kv_strides, 0};
    DLTensor v_arg = {v->data, device, 3, fp8, kv_shape, kv_strides, 0};
    DLTensor o_arg = {output, device, 3, {kDLBfloat, 16, 1}, q_shape, q_strides, 0};
    DLTensor cu_q = {seq_lens, device, 1, {kDLInt, 32, 1}, cu_shape, cu_strides, 0};
    DLTensor cu_k = {seq_lens + 2, device, 1, {kDLInt, 32, 1}, cu_shape, cu_strides, 0};
    const float scale_log2 = ggml_get_op_params_f32(dst, 0)*1.4426950408889634f;
    flashinfer_stream_guard stream_guard(ctx.device, ctx.stream());
    run(&q_arg, &k_arg, &v_arg, &o_arg, &cu_q, &cu_k, scale_log2, n_queries);
    ggml_get_to_fp32_cuda(GGML_TYPE_BF16)(output, (float *) dst->data, n, ctx.stream());
    return true;
}
#endif
