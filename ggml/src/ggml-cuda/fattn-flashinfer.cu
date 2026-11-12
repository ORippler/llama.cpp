#ifdef GGML_CUDA_FLASHINFER
#include "fattn-flashinfer.cuh"
#include "convert.cuh"
#include "nvfp4-flashinfer.cuh"
#include "quantize.cuh"

#include <mutex>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

struct ggml_cuda_flashinfer_state {
    tvm::ffi::Optional<tvm::ffi::Module> module;
    tvm::ffi::Function run;
    tvm::ffi::Function gemm;

    ggml_cuda_flashinfer_state() {
        const char * path = std::getenv("GGML_CUDA_FLASHINFER_LIBRARY");
        try {
            module = tvm::ffi::Module::LoadFromFile(path ? path : GGML_CUDA_FLASHINFER_LIBRARY);
            auto function = module.value()->GetFunction("ggml_flashinfer_fp8_v1");
            if (function.has_value()) {
                run = function.value();
            }
            auto g = module.value()->GetFunction("ggml_flashinfer_nvfp4_gemm_v1");
            if (g.has_value()) {
                gemm = g.value();
            }
            GGML_LOG_INFO("ggml_cuda_flashinfer: loaded TVM FFI library (attention=%d, gemm=%d)\n", run.defined(), gemm.defined());
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
        ctx.flashinfer = flashinfer_state();
    }

    if (!ctx.flashinfer->run.defined()) {
        return false;
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

    int64_t q_shape[] = {n_queries, 24, 256};
    int64_t kv_shape[] = {n_kv, 4, 256};
    int64_t q_strides[] = {24*256, 256, 1};
    int64_t kv_strides[] = {4*256, 256, 1};
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
    ctx.flashinfer->run(&q_arg, &k_arg, &v_arg, &o_arg, &cu_q, &cu_k, scale_log2, n_queries);
    ggml_get_to_fp32_cuda(GGML_TYPE_BF16)(output, (float *) dst->data, n, ctx.stream());
    return true;
}

static __global__ void flashinfer_row_scale(float * output, const nv_bfloat16 * source, const float * scales,
        const nv_bfloat16 * gate, const float * up_scale, const float * gate_scale, int64_t n, int64_t count) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float value = __bfloat162float(source[i]) * scales[i / n];
        if (gate) {
            value *= up_scale ? *up_scale : 1.0f;
            const float g = __bfloat162float(gate[i]) * scales[i / n] * (gate_scale ? *gate_scale : 1.0f);
            value *= g / (1.0f + expf(-g));
        }
        output[i] = value;
    }
}

static __global__ void flashinfer_set_one(float * value) {
    *value = 1.0f;
}

static bool flashinfer_mul_mat(ggml_backend_cuda_context & ctx,
        const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * dst, const ggml_tensor * precision_node,
        const ggml_tensor * gate_weight = nullptr, const ggml_tensor * up_scale = nullptr, const ggml_tensor * gate_scale = nullptr) {
    static const bool enabled = std::getenv("GGML_CUDA_FLASHINFER_GEMM") != nullptr;
    const auto layout = ggml_cuda_nvfp4_get_layout(weight);
    if (!enabled || !layout.data || weight->view_src || layout.experts != 1 ||
            ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_BLACKWELL ||
            input->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            input->ne[0] != layout.k || input->ne[1] < 128 || input->ne[2] != 1 || input->ne[3] != 1 ||
            input->nb[0] != sizeof(float) || !ggml_is_contiguous(dst) ||
            dst->ne[0] != layout.n || dst->ne[1] != input->ne[1] || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            !ggml_cuda_nvfp4_q4_allowed(precision_node)) {
        return false;
    }
    const auto gate = ggml_cuda_nvfp4_get_layout(gate_weight);
    if (gate_weight && (!gate.data || gate_weight->view_src || gate.k != layout.k || gate.n != layout.n || gate.experts != 1)) {
        return false;
    }
    if (!ctx.flashinfer) {
        ctx.flashinfer = flashinfer_state();
    }
    if (!ctx.flashinfer->gemm.defined()) {
        return false;
    }
    const int64_t m = input->ne[1], n = layout.n, k = layout.k;
    const int64_t m_padded = GGML_PAD(m, 128);
    ggml_cuda_pool_alloc<uint8_t> a(ctx.pool(), m_padded * k / 2);
    ggml_cuda_pool_alloc<uint8_t> a_sf(ctx.pool(), GGML_PAD(m, 128) * k / 16);
    ggml_cuda_pool_alloc<float> row_scales(ctx.pool(), m + 1);
    ggml_cuda_pool_alloc<nv_bfloat16> padded_output(ctx.pool(), m_padded * n);
    ggml_cuda_pool_alloc<nv_bfloat16> gate_output(ctx.pool());
    if (gate_weight) {
        gate_output.alloc(m_padded * n);
    }
    nv_bfloat16 * output = padded_output.ptr;
    if (m != m_padded) {
        CUDA_CHECK(cudaMemsetAsync(a.ptr + m * k / 2, 0, (m_padded - m) * k / 2, ctx.stream()));
    }
    quantize_flashinfer_nvfp4_cuda((const float *) input->data, a.ptr, a_sf.ptr, row_scales.ptr,
        k, m, input->nb[1] / sizeof(float), ctx.stream());
    flashinfer_set_one<<<1, 1, 0, ctx.stream()>>>(row_scales.ptr + m);

    int64_t a_shape[] = {m_padded, k / 2}, b_shape[] = {n, k / 2}, out_shape[] = {m_padded, n};
    int64_t a_strides[] = {k / 2, 1}, out_strides[] = {n, 1}, stride_one[] = {1};
    int64_t asf_shape[] = {GGML_PAD(m, 128) * k / 16}, bsf_shape[] = {n * k / 16}, alpha_shape[] = {1};
    const DLDevice device = {kDLCUDA, ctx.device};
    const DLDataType bytes = {kDLUInt, 8, 1}, f32 = {kDLFloat, 32, 1};
    DLTensor a_arg = {a.ptr, device, 2, bytes, a_shape, a_strides, 0};
    DLTensor b_arg = {layout.data, device, 2, bytes, b_shape, a_strides, 0};
    DLTensor asf_arg = {a_sf.ptr, device, 1, bytes, asf_shape, stride_one, 0};
    DLTensor bsf_arg = {layout.scales, device, 1, bytes, bsf_shape, stride_one, 0};
    DLTensor out_arg = {output, device, 2, {kDLBfloat, 16, 1}, out_shape, out_strides, 0};
    DLTensor alpha_arg = {row_scales.ptr + m, device, 1, f32, alpha_shape, stride_one, 0};
    flashinfer_stream_guard guard(ctx.device, ctx.stream());
    ctx.flashinfer->gemm(&a_arg, &b_arg, &asf_arg, &bsf_arg, &out_arg, &alpha_arg);
    if (gate_weight) {
        b_arg.data = gate.data;
        bsf_arg.data = gate.scales;
        out_arg.data = gate_output.ptr;
        ctx.flashinfer->gemm(&a_arg, &b_arg, &asf_arg, &bsf_arg, &out_arg, &alpha_arg);
    }
    flashinfer_row_scale<<<(m * n + 255) / 256, 256, 0, ctx.stream()>>>((float *) dst->data, output, row_scales.ptr,
        gate_output.ptr, up_scale ? (const float *) up_scale->data : nullptr, gate_scale ? (const float *) gate_scale->data : nullptr, n, m * n);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_flashinfer_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * dst) {
    return flashinfer_mul_mat(ctx, weight, input, dst, dst);
}

bool ggml_cuda_flashinfer_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * up_scale, const ggml_tensor * gate_scale, ggml_tensor * dst) {
    if (up->src[1] != gate->src[1] || !ggml_cuda_nvfp4_q4_allowed(gate)) {
        return false;
    }
    for (const ggml_tensor * scale : {up_scale, gate_scale}) {
        if (scale && (scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1)) {
            return false;
        }
    }
    return flashinfer_mul_mat(ctx, up->src[0], up->src[1], dst, up, gate->src[0], up_scale, gate_scale);
}

static __global__ void flashinfer_moe_scales(float * output, const float * up, const float * gate,
        const float * down, const float * input_scale, const float * down_input_scale, int experts) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < experts) {
        output[e] = input_scale[e];
        output[experts + 2 * e] = up[e] * input_scale[e];
        output[experts + 2 * e + 1] = gate[e] * input_scale[e];
        output[3 * experts + e] = down[e] * down_input_scale[e];
        output[4 * experts + e] = down_input_scale[e];
    }
}

bool ggml_cuda_flashinfer_moe(ggml_backend_cuda_context & ctx, const ggml_cuda_flashinfer_moe_args & args) {
    static const bool enabled = std::getenv("GGML_CUDA_FLASHINFER_MOE") != nullptr;
    if (!enabled || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_BLACKWELL) {
        return false;
    }
    const auto up = ggml_cuda_nvfp4_get_layout(args.up);
    const auto gate = ggml_cuda_nvfp4_get_layout(args.gate);
    const auto down = ggml_cuda_nvfp4_get_layout(args.down);
    if (!up.data || !gate.data || !down.data || up.k != 2048 || up.n != 512 || up.experts != 256 ||
            gate.k != up.k || gate.n != up.n || gate.experts != up.experts ||
            down.k != 512 || down.n != 2048 || down.experts != 256 ||
            args.up->view_src || args.gate->view_src || args.down->view_src ||
            args.input->type != GGML_TYPE_F32 || args.ids->type != GGML_TYPE_I32 || args.weights->type != GGML_TYPE_F32 ||
            args.dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(args.input) || args.ids->nb[0] != sizeof(int32_t) ||
            !ggml_is_contiguous(args.weights) || !ggml_is_contiguous(args.dst)) {
        return false;
    }
    const int m = args.ids->ne[1];
    if (args.ids->ne[0] != 8 || args.ids->ne[2] != 1 || args.ids->ne[3] != 1 || m < 128 || m > 2048 ||
            ggml_nelements(args.input) != int64_t(m) * 2048 || ggml_nelements(args.weights) != int64_t(m) * 8 ||
            args.dst->ne[0] != 2048 || args.dst->ne[1] != m || args.dst->ne[2] != 1 || args.dst->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * us = ggml_cuda_nvfp4_get_aux(args.up, ".scale");
    const ggml_tensor * gs = ggml_cuda_nvfp4_get_aux(args.gate, ".scale");
    const ggml_tensor * ds = ggml_cuda_nvfp4_get_aux(args.down, ".scale");
    const ggml_tensor * is = ggml_cuda_nvfp4_get_aux(args.up, ".input_scale");
    const ggml_tensor * dis = ggml_cuda_nvfp4_get_aux(args.down, ".input_scale");
    for (const ggml_tensor * t : {us, gs, ds, is, dis}) {
        if (!t || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t) != 256) {
            return false;
        }
    }
    if (!ctx.flashinfer) {
        ctx.flashinfer = flashinfer_state();
    }
    if (!ctx.flashinfer->module.has_value()) {
        return false;
    }
    const std::string symbol = "ggml_flashinfer_nvfp4_moe_sm" + std::to_string(ggml_cuda_info().devices[ctx.device].nsm) + "_v2";
    auto function = ctx.flashinfer->module.value()->GetFunction(symbol);
    if (!function.has_value()) {
        return false;
    }

    constexpr int e = 256, k = 2048, n = 512;
    const int tiles = (m * 8 + 127) / 128 + e - 1;
    // The dynamic scheduler can split each I512 tile into two I256 tasks.
    const int tasks = 2 * tiles;
    const int rows = tiles * 128;
    ggml_cuda_pool_alloc<int32_t> ids(ctx.pool(), m * 8);
    ggml_cuda_pool_alloc<nv_bfloat16> input(ctx.pool(), m * k);
    ggml_cuda_pool_alloc<nv_bfloat16> output(ctx.pool(), m * k);
    ggml_cuda_pool_alloc<uint8_t> packed_input(ctx.pool(), size_t(rows) * k / 2);
    ggml_cuda_pool_alloc<uint8_t> input_sf(ctx.pool(), size_t(rows) * k / 16);
    ggml_cuda_pool_alloc<int32_t> state(ctx.pool(), 6 + 3 * e + 2 * tasks + rows);
    ggml_cuda_pool_alloc<float> token_weights(ctx.pool(), rows);
    ggml_cuda_pool_alloc<float> scales(ctx.pool(), 5 * e);
    const cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemcpy2DAsync(ids.ptr, 8 * sizeof(int32_t), args.ids->data, args.ids->nb[1],
        8 * sizeof(int32_t), m, cudaMemcpyDeviceToDevice, stream));
    ggml_get_to_bf16_cuda(GGML_TYPE_F32)(args.input->data, input.ptr, int64_t(m) * k, stream);
    CUDA_CHECK(cudaMemsetAsync(state.ptr, 0, (6 + 3 * e + 2 * tasks + rows) * sizeof(int32_t), stream));
    flashinfer_moe_scales<<<1, e, 0, stream>>>(scales.ptr, (const float *) us->data, (const float *) gs->data,
        (const float *) ds->data, (const float *) is->data, (const float *) dis->data, e);

    const DLDevice device = {kDLCUDA, ctx.device};
    const DLDataType f32 = {kDLFloat, 32, 1}, i32 = {kDLInt, 32, 1}, fp4 = {kDLFloat4_e2m1fn, 4, 2};
    int64_t w13_shape[] = {n, k / 2, e}, w13_strides[] = {k / 2, 1, n * k / 2};
    int64_t down_shape[] = {k, n / 2, e}, down_strides[] = {n / 2, 1, k * n / 2};
    int64_t one_shape[] = {1}, expert_shape[] = {e}, bases_shape[] = {e + 1}, alpha_shape[] = {2 * e}, strides[] = {1};
    DLTensor w13_arg = {up.data, device, 3, fp4, w13_shape, w13_strides, 0};
    DLTensor gate_arg = {gate.data, device, 3, fp4, w13_shape, w13_strides, 0};
    DLTensor down_arg = {down.data, device, 3, fp4, down_shape, down_strides, 0};
    DLTensor barrier = {state.ptr, device, 1, i32, one_shape, strides, 0};
    DLTensor epoch = {state.ptr + 1, device, 1, i32, one_shape, strides, 0};
    DLTensor pair_head = {state.ptr + 2, device, 1, i32, one_shape, strides, 0};
    DLTensor task_head = {state.ptr + 3, device, 1, i32, one_shape, strides, 0};
    DLTensor task_tail = {state.ptr + 4, device, 1, i32, one_shape, strides, 0};
    DLTensor row_counts = {state.ptr + 5, device, 1, i32, expert_shape, strides, 0};
    DLTensor write_rows = {state.ptr + 5 + e, device, 1, i32, expert_shape, strides, 0};
    DLTensor tile_base = {state.ptr + 5 + 2 * e, device, 1, i32, bases_shape, strides, 0};
    DLTensor input_gs = {scales.ptr, device, 1, f32, expert_shape, strides, 0};
    DLTensor alpha = {scales.ptr + e, device, 1, f32, alpha_shape, strides, 0};
    DLTensor down_alpha = {scales.ptr + 3 * e, device, 1, f32, expert_shape, strides, 0};
    DLTensor global_scale = {scales.ptr + 4 * e, device, 1, f32, expert_shape, strides, 0};
    int32_t * task_expert = state.ptr + 6 + 3 * e;
    int32_t * valid_rows = task_expert + tasks;
    int32_t * token_map = valid_rows + tasks;
    const auto address = [](const void * ptr) { return reinterpret_cast<int64_t>(ptr); };
    flashinfer_stream_guard guard(ctx.device, stream);
    function.value()(address(input.ptr), address(ids.ptr), address(args.weights->data),
        address(packed_input.ptr), address(input_sf.ptr), address(packed_input.ptr), address(input_sf.ptr),
        &barrier, &epoch, &pair_head, &task_head, &task_tail, address(task_expert), address(valid_rows),
        &w13_arg, address(up.scales), &gate_arg, address(gate.scales), &down_arg, address(down.scales), &row_counts, &write_rows, &tile_base,
        &input_gs, &alpha, &down_alpha, &global_scale, address(output.ptr), address(token_map), address(token_weights.ptr),
        m, rows, rows, tasks);
    ggml_get_to_fp32_cuda(GGML_TYPE_BF16)(output.ptr, (float *) args.dst->data, int64_t(m) * k, stream);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#endif
