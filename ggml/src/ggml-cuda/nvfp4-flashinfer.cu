#include "nvfp4-flashinfer.cuh"
#include "unary.cuh"

#include <cctype>

bool ggml_cuda_nvfp4_q4_allowed(const ggml_tensor * node) {
    static const std::string precision = [] {
        const char * value = std::getenv("GGML_CUDA_MMQ_PREC");
        std::string result = value ? value : "auto";
        for (char & c : result) {
            c = std::tolower((unsigned char) c);
        }
        return result;
    }();
    return precision == "q4" || (precision != "q8" && ggml_get_op_params_i32(node, 3) != GGML_PREC_Q8);
}

#ifndef GGML_CUDA_FLASHINFER
bool ggml_cuda_flashinfer_mul_mat_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * matmul, const ggml_tensor * scale, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, matmul, scale, dst);
    return false;
}

bool ggml_cuda_flashinfer_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * up_scale, const ggml_tensor * gate_scale, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, up, gate, up_scale, gate_scale, dst);
    return false;
}

bool ggml_cuda_flashinfer_moe(ggml_backend_cuda_context & ctx, const ggml_cuda_flashinfer_moe_args & args) {
    GGML_UNUSED_VARS(ctx, args);
    return false;
}

bool ggml_cuda_flashinfer_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, weight, input, dst);
    return false;
}
#endif

bool ggml_cuda_nvfp4_repack_enabled(int device) {
#ifdef GGML_CUDA_FLASHINFER
    static const bool enabled = std::getenv("GGML_CUDA_FLASHINFER_REPACK") != nullptr;
    return enabled && ggml_cuda_info().devices[device].cc == GGML_CUDA_CC_BLACKWELL;
#else
    GGML_UNUSED(device);
    return false;
#endif
}

bool ggml_cuda_nvfp4_repack_candidate(const ggml_tensor * tensor) {
    return tensor->type == GGML_TYPE_NVFP4 && tensor->op == GGML_OP_NONE && !tensor->view_src &&
        ggml_is_contiguous(tensor) && tensor->ne[0] > 0 && tensor->ne[0] % 64 == 0 &&
        tensor->ne[1] > 0 && tensor->ne[1] % 128 == 0 && tensor->ne[2] > 0 && tensor->ne[3] == 1;
}

static __device__ void nvfp4_store_bits(uint8_t * dst, uint8_t value, uint8_t mask) {
    auto * word = (unsigned int *) ((uintptr_t) dst & ~uintptr_t(3));
    const int shift = ((uintptr_t) dst & 3) * 8;
    const unsigned int bits = (unsigned int) mask << shift;
    unsigned int old = atomicCAS(word, 0, 0);
    unsigned int expected;
    do {
        expected = old;
        old = atomicCAS(word, expected, (expected & ~bits) | ((unsigned int) (value & mask) << shift));
    } while (old != expected);
}

static __global__ void nvfp4_pack_bytes(ggml_cuda_nvfp4_layout layout, const uint8_t * src, size_t offset, size_t size) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    const size_t pos = offset + i;
    const size_t block = pos / sizeof(block_nvfp4);
    const int field = pos % sizeof(block_nvfp4);
    const int64_t row = block / (layout.k / 64);
    const int64_t expert = row / layout.n;
    const int64_t kblock = block % (layout.k / 64);
    if (field < 4) {
        const size_t sf = expert * layout.n * layout.k / 16 +
            ggml_cuda_nvfp4_scale_index(row % layout.n, kblock * 4 + field, layout.k, layout.n);
        layout.scales[sf] = src[i];
    } else {
        const int j = (field - 4) % 8;
        const size_t first = row * (layout.k / 2) + kblock * 32 + (field - 4) / 8 * 8 + j / 2;
        const int shift = (j % 2) * 4;
        nvfp4_store_bits(layout.data + first, (src[i] & 15) << shift, 15 << shift);
        nvfp4_store_bits(layout.data + first + 4, (src[i] >> 4) << shift, 15 << shift);
    }
}

static __global__ void nvfp4_unpack_bytes(ggml_cuda_nvfp4_layout layout, uint8_t * dst, size_t offset, size_t size) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    const size_t pos = offset + i;
    const size_t block = pos / sizeof(block_nvfp4);
    const int field = pos % sizeof(block_nvfp4);
    const int64_t row = block / (layout.k / 64);
    const int64_t expert = row / layout.n;
    const int64_t kblock = block % (layout.k / 64);
    if (field < 4) {
        dst[i] = layout.scales[expert * layout.n * layout.k / 16 +
            ggml_cuda_nvfp4_scale_index(row % layout.n, kblock * 4 + field, layout.k, layout.n)];
    } else {
        const int j = (field - 4) % 8;
        const size_t first = row * (layout.k / 2) + kblock * 32 + (field - 4) / 8 * 8 + j / 2;
        const int shift = (j % 2) * 4;
        dst[i] = ((layout.data[first] >> shift) & 15) | (((layout.data[first + 4] >> shift) & 15) << 4);
    }
}

static __global__ void nvfp4_pack_blocks(ggml_cuda_nvfp4_layout layout, const block_nvfp4 * src, size_t first, size_t count) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const size_t block = first + i;
    const int64_t row = block / (layout.k / 64);
    const int64_t kblock = block % (layout.k / 64);
    uint8_t * data = layout.data + block * 32;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        layout.scales[(row / layout.n) * layout.n * layout.k / 16 +
            ggml_cuda_nvfp4_scale_index(row % layout.n, kblock * 4 + s, layout.k, layout.n)] = src[i].d[s];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int shift = j < 4 ? 0 : 4;
            data[s * 8 + j] = ((src[i].qs[s * 8 + (2 * j) % 8] >> shift) & 15) |
                (((src[i].qs[s * 8 + (2 * j + 1) % 8] >> shift) & 15) << 4);
        }
    }
}

void ggml_cuda_nvfp4_pack(const ggml_cuda_nvfp4_layout & layout, const void * src, size_t offset, size_t size, cudaStream_t stream) {
    const auto * data = (const uint8_t *) src;
    const size_t prefix = std::min(size, (sizeof(block_nvfp4) - offset % sizeof(block_nvfp4)) % sizeof(block_nvfp4));
    const size_t blocks = (size - prefix) / sizeof(block_nvfp4);
    const size_t tail = prefix + blocks * sizeof(block_nvfp4);
    if (prefix) {
        nvfp4_pack_bytes<<<1, 256, 0, stream>>>(layout, data, offset, prefix);
    }
    if (blocks) {
        nvfp4_pack_blocks<<<(blocks + 255) / 256, 256, 0, stream>>>(layout,
            (const block_nvfp4 *) (data + prefix), (offset + prefix) / sizeof(block_nvfp4), blocks);
    }
    if (tail < size) {
        nvfp4_pack_bytes<<<1, 256, 0, stream>>>(layout, data + tail, offset + tail, size - tail);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_nvfp4_set(const ggml_cuda_nvfp4_layout & layout, const void * src, size_t offset, size_t size, cudaStream_t stream) {
    const size_t chunk_size = std::min<size_t>(size, 8 * 1024 * 1024);
    if (!chunk_size) {
        return;
    }
    uint8_t * staging;
    CUDA_CHECK(cudaMalloc(&staging, chunk_size));
    for (size_t i = 0; i < size; i += chunk_size) {
        const size_t n = std::min(chunk_size, size - i);
        CUDA_CHECK(cudaMemcpyAsync(staging, (const char *) src + i, n, cudaMemcpyHostToDevice, stream));
        ggml_cuda_nvfp4_pack(layout, staging, offset + i, n, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(staging));
}

void ggml_cuda_nvfp4_unpack(const ggml_cuda_nvfp4_layout & layout, void * dst, size_t offset, size_t size, cudaStream_t stream) {
    if (size) {
        nvfp4_unpack_bytes<<<(size + 255) / 256, 256, 0, stream>>>(layout, (uint8_t *) dst, offset, size);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_nvfp4_get(const ggml_cuda_nvfp4_layout & layout, void * dst, size_t offset, size_t size, cudaStream_t stream) {
    const size_t chunk_size = std::min<size_t>(size, 8 * 1024 * 1024);
    if (!chunk_size) {
        return;
    }
    uint8_t * staging;
    CUDA_CHECK(cudaMalloc(&staging, chunk_size));
    for (size_t i = 0; i < size; i += chunk_size) {
        const size_t n = std::min(chunk_size, size - i);
        ggml_cuda_nvfp4_unpack(layout, staging, offset + i, n, stream);
        CUDA_CHECK(cudaMemcpyAsync((char *) dst + i, staging, n, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaFree(staging));
}

ggml_cuda_nvfp4_unpacked::ggml_cuda_nvfp4_unpacked(ggml_backend_cuda_context & ctx, const ggml_tensor * src) :
    allocation(ctx.pool()), tensor(*src) {
    const auto layout = ggml_cuda_nvfp4_get_layout(src);
    if (!layout.data) {
        return;
    }
    const size_t size = ggml_nbytes(src);
    allocation.alloc(GGML_PAD(size, 512) + 512);
    tensor.data = allocation.ptr;
    tensor.extra = nullptr;
    tensor.view_src = nullptr;
    const size_t offset = (const char *) src->data - (const char *) layout.data;
    ggml_cuda_nvfp4_unpack(layout, tensor.data, offset, size, ctx.stream());
    CUDA_CHECK(cudaMemsetAsync((char *) tensor.data + size, 0, GGML_PAD(size, 512) + 512 - size, ctx.stream()));
}

static __device__ __forceinline__ float4 nvfp4_to_float4(uint16_t bits) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
    const half2 low = __nv_cvt_fp4x2_to_halfraw2(bits & 255, __NV_E2M1);
    const half2 high = __nv_cvt_fp4x2_to_halfraw2(bits >> 8, __NV_E2M1);
    const float2 a = __half22float2(low), b = __half22float2(high);
    return make_float4(a.x, a.y, b.x, b.y);
#else
    return make_float4(0.5f * kvalues_fp4[bits & 15], 0.5f * kvalues_fp4[(bits >> 4) & 15],
        0.5f * kvalues_fp4[(bits >> 8) & 15], 0.5f * kvalues_fp4[(bits >> 12) & 15]);
#endif
}

static __global__ void nvfp4_mul_mat_vec(ggml_cuda_nvfp4_layout weight, ggml_cuda_nvfp4_layout gate,
        const float * input, const int32_t * ids, float * dst, int64_t input_stride, int input_columns,
        const float * weight_scale, const float * gate_scale, ggml_glu_op glu_op, float glu_limit) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (row >= weight.n) {
        return;
    }
    const int expert = ids ? ids[blockIdx.y] : 0;
    const float * x = input + (blockIdx.y % input_columns) * input_stride;
    const uint8_t * w = weight.data + ((int64_t) expert * weight.n + row) * (weight.k / 2);
    const uint8_t * g = gate.data ? gate.data + ((int64_t) expert * gate.n + row) * (gate.k / 2) : nullptr;
    float sum = 0.0f, gate_sum = 0.0f;
    for (int k = lane * 16; k < weight.k; k += WARP_SIZE * 16) {
        const size_t sf_index = (int64_t) expert * weight.n * weight.k / 16 +
            ggml_cuda_nvfp4_scale_index(row, k / 16, weight.k, weight.n);
        const float d = 2.0f * ggml_cuda_ue4m3_to_fp32(weight.scales[sf_index]);
        const float gd = g ? 2.0f * ggml_cuda_ue4m3_to_fp32(gate.scales[sf_index]) : 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 values = *(const float4 *) (x + k + 4 * j);
            const float4 ws = nvfp4_to_float4(*(const uint16_t *) (w + k / 2 + 2 * j));
            sum = fmaf(d * ws.x, values.x, sum);
            sum = fmaf(d * ws.y, values.y, sum);
            sum = fmaf(d * ws.z, values.z, sum);
            sum = fmaf(d * ws.w, values.w, sum);
            if (g) {
                const float4 gs = nvfp4_to_float4(*(const uint16_t *) (g + k / 2 + 2 * j));
                gate_sum = fmaf(gd * gs.x, values.x, gate_sum);
                gate_sum = fmaf(gd * gs.y, values.y, gate_sum);
                gate_sum = fmaf(gd * gs.z, values.z, gate_sum);
                gate_sum = fmaf(gd * gs.w, values.w, gate_sum);
            }
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    gate_sum = warp_reduce_sum<WARP_SIZE>(gate_sum);
    if (lane == 0) {
        sum *= weight_scale ? weight_scale[expert] : 1.0f;
        if (g) {
            gate_sum *= gate_scale ? gate_scale[expert] : 1.0f;
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU: sum *= ggml_cuda_op_silu_single(gate_sum); break;
                case GGML_GLU_OP_GEGLU: sum *= ggml_cuda_op_gelu_single(gate_sum); break;
                case GGML_GLU_OP_SWIGLU_OAI: sum = ggml_cuda_op_swiglu_oai_single(gate_sum, sum); break;
                case GGML_GLU_OP_SWIGLU_CLAMP: sum = ggml_cuda_op_swiglu_clamp_single(gate_sum, sum, glu_limit); break;
                default: sum *= gate_sum; break;
            }
        }
        dst[(int64_t) blockIdx.y * weight.n + row] = sum;
    }
}

bool ggml_cuda_nvfp4_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * weight, const ggml_tensor * input,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    const auto layout = ggml_cuda_nvfp4_get_layout(weight);
    const auto gate = fusion ? ggml_cuda_nvfp4_get_layout(fusion->gate) : ggml_cuda_nvfp4_layout{};
    if (!layout.data || weight->view_src || input->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(input) || !ggml_is_contiguous(dst) || !ggml_cuda_is_aligned(input, 16) ||
            input->ne[0] != layout.k || dst->ne[0] != layout.n ||
            input->ne[2] != 1 || input->ne[3] != 1 || (!ids && input->ne[1] != 1) ||
            (!ids && layout.experts != 1) ||
            (ids && (ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || ids->nb[0] != sizeof(int32_t))) ||
            (fusion && (fusion->x_bias || fusion->gate_bias || fusion->shared_up ||
                (fusion->gate && (!gate.data || fusion->gate->view_src || gate.k != layout.k || gate.n != layout.n || gate.experts != layout.experts))))) {
        return false;
    }
    const dim3 grid((layout.n + 3) / 4, ids ? ids->ne[0] : 1, 1);
    nvfp4_mul_mat_vec<<<grid, 4 * WARP_SIZE, 0, ctx.stream()>>>(layout, gate, (const float *) input->data,
        ids ? (const int32_t *) ids->data : nullptr, (float *) dst->data, input->nb[1] / sizeof(float), input->ne[1],
        fusion && fusion->x_scale ? (const float *) fusion->x_scale->data : nullptr,
        fusion && fusion->gate_scale ? (const float *) fusion->gate_scale->data : nullptr,
        fusion ? fusion->glu_op : GGML_GLU_OP_SWIGLU, fusion ? fusion->glu_limit : 0.0f);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
