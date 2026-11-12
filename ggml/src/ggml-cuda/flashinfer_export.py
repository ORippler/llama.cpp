import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


FLASHINFER_COMMIT = "77039051b0be921788dc2784ec41423b558878ec"


def main():
    parser = argparse.ArgumentParser(description="Export SM120/SM121 FlashInfer kernels to a TVM FFI library")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cc", default="cc", help="Host linker driver")
    parser.add_argument("--gpu-arch", choices=("sm_120a", "sm_121a", "sm_120f"), default="sm_120a")
    parser.add_argument("--kernels", choices=("attention", "gemm", "moe", "all"), default="attention")
    parser.add_argument("--max-active-clusters", type=int, default=128)
    parser.add_argument("--moe-sm-counts", type=int, nargs="+", help="MoE worker-grid variants, e.g. 48 82; default: visible GPU")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.source_dir), "rev-parse", "HEAD"], text=True).strip()
    if revision != FLASHINFER_COMMIT:
        parser.error(f"FlashInfer must be checked out at {FLASHINFER_COMMIT}, found {revision}")
    if subprocess.run(["git", "-C", str(args.source_dir), "diff", "--quiet", "HEAD", "--", "flashinfer"]).returncode:
        parser.error("FlashInfer sources must match the pinned checkout")
    if args.max_active_clusters <= 0:
        parser.error("--max-active-clusters must be positive")
    if args.moe_sm_counts and any(count <= 0 for count in args.moe_sm_counts):
        parser.error("--moe-sm-counts must be positive")
    sys.path.insert(0, str(args.source_dir.resolve()))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(args.output_dir.resolve()))
    os.environ["CUTE_DSL_ARCH"] = args.gpu_arch

    import cutlass.cute as cute  # ty: ignore[unresolved-import]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects = []
    if args.kernels in ("attention", "all"):
        for q_heads, kv_heads in ((24, 4), (16, 2)):
            objects.append(export_attention(args.output_dir, q_heads, kv_heads, args.gpu_arch))
    if args.kernels in ("gemm", "all"):
        for tile_m in (128, 64):
            objects.append(export_gemm(args.output_dir, args.max_active_clusters, tile_m, args.gpu_arch))
        for m in (1, 2, 4, 8, 16):
            for n, k, splits in ((17408, 5120, 1), (5120, 17408, 2), (248320, 5120, 1)):
                objects.append(export_w4a16(args.output_dir, m, n, k, splits, args.gpu_arch))
    if args.kernels in ("moe", "all"):
        objects.extend(export_moe(args.source_dir, args.output_dir, args.gpu_arch, args.moe_sm_counts))

    libraries = [Path(lib) for lib in cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)]
    for library in libraries:
        target = args.output_dir / library.name
        if library.resolve() != target.resolve():
            shutil.copy2(library, target)
    subprocess.run([args.cc, "-shared", "-o", str(args.output_dir / "libggml_flashinfer_fp8.so"),
                    *map(str, objects), f"-L{args.output_dir}", "-Wl,-rpath,$ORIGIN",
                    *[f"-l:{lib.name}" for lib in libraries]], check=True)


def export_attention(output_dir, q_heads, kv_heads, gpu_arch):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    from cuda.bindings import driver as cuda
    from flashinfer.cute_dsl.attention.fmha.sm120.fmha_prefill_fp8_tma import SM120FusedMultiHeadAttentionFP8ForwardTMA

    fmha = SM120FusedMultiHeadAttentionFP8ForwardTMA(
        in_dtype=cutlass.Float8E4M3FN, out_dtype=cutlass.BFloat16,
        is_causal=True, head_tile=256, kv_tile=128, q_tile=128,
        use_paged_kv=False, balanced_scheduler=True)

    @cute.jit
    def launch(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, o: cute.Tensor,
               cu_q: cute.Tensor, cu_k: cute.Tensor, scale: cutlass.Float32,
               max_q: cutlass.Int32, stream: cuda.CUstream):
        fmha(q, k, v, o, None, scale, cutlass.Float32(1.0), None, stream,
             None, cu_q, None, cu_k, max_q, False)

    q_len, k_len = cute.sym_int(), cute.sym_int()
    q = make_fake_compact_tensor(cutlass.Float8E4M3FN, (q_len, q_heads, 256), stride_order=(2, 1, 0), assumed_align=16)
    k = make_fake_compact_tensor(cutlass.Float8E4M3FN, (k_len, kv_heads, 256), stride_order=(2, 1, 0), assumed_align=16)
    o = make_fake_compact_tensor(cutlass.BFloat16, (q_len, q_heads, 256), stride_order=(2, 1, 0), assumed_align=16)
    cu = make_fake_compact_tensor(cutlass.Int32, (2,), assumed_align=4)
    compiled = cute.compile(launch, q, k, k, o, cu, cu, cutlass.Float32(1.0), cutlass.Int32(1),
                            make_fake_stream(use_tvm_ffi_env_stream=True), options=f"--gpu-arch {gpu_arch} --enable-tvm-ffi")
    symbol = "ggml_flashinfer_fp8_v1" if (q_heads, kv_heads) == (24, 4) else "ggml_flashinfer_fp8_h16_kv2_v1"
    object_path = output_dir / f"{symbol}.o"
    compiled.export_to_c(str(object_path), function_name=symbol)
    return object_path


def export_gemm(output_dir, max_active_clusters, tile_m, gpu_arch):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    from cuda.bindings import driver as cuda
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import DenseGemmKernel

    gemm = DenseGemmKernel(16, (tile_m, 128), (1, 1), enable_pdl=False)

    @cute.jit
    def launch(a: cute.Tensor, b: cute.Tensor, a_sf: cute.Tensor, b_sf: cute.Tensor,
               output: cute.Tensor, alpha: cute.Tensor, stream: cuda.CUstream):
        gemm.wrapper(a, b, output, (a.shape[0] + 127) // 128, (b.shape[0] + 127) // 128,
                     (a.shape[1] + 31) // 32, 1,
                     cute.recast_ptr(a_sf.iterator, dtype=cutlass.Float8E4M3FN),
                     cute.recast_ptr(b_sf.iterator, dtype=cutlass.Float8E4M3FN),
                     alpha, max_active_clusters, stream)

    m, n, k = cute.sym_int(), cute.sym_int(), cute.sym_int(divisibility=32)
    a = make_fake_compact_tensor(cutlass.Uint8, (m, k), stride_order=(1, 0), assumed_align=32)
    b = make_fake_compact_tensor(cutlass.Uint8, (n, k), stride_order=(1, 0), assumed_align=32)
    a_sf = make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16)
    b_sf = make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16)
    output = make_fake_compact_tensor(cutlass.BFloat16, (m, n), stride_order=(1, 0), assumed_align=16)
    alpha = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    compiled = cute.compile(launch, a, b, a_sf, b_sf, output, alpha,
                            make_fake_stream(use_tvm_ffi_env_stream=True),
                            options=f"--gpu-arch {gpu_arch} --enable-tvm-ffi")
    object_path = output_dir / f"ggml_flashinfer_nvfp4_{tile_m}.o"
    symbol = "ggml_flashinfer_nvfp4_gemm_v1" if tile_m == 128 else "ggml_flashinfer_nvfp4_gemm_m64_v1"
    compiled.export_to_c(str(object_path), function_name=symbol)
    return object_path


def export_w4a16(output_dir, m, n, k, splits, gpu_arch):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    from cuda.bindings import driver as cuda
    from flashinfer.gemm.kernels.native_bf16_fp4.staged_kernel import NativeBf16Fp4StagedKernel

    warps = 8 if n == 248320 and m == 1 else 4
    stages = 2 if n == 17408 or (n == 248320 and m == 4) else 3
    gemm = NativeBf16Fp4StagedKernel(splits, warps, False, 128, stages)

    @cute.jit
    def launch(a: cute.Tensor, b: cute.Tensor, sf: cute.Tensor, output: cute.Tensor,
               alpha: cute.Tensor, partial: cute.Tensor, stream: cuda.CUstream):
        a_view = cute.make_tensor(cute.recast_ptr(a.iterator, dtype=cutlass.Int32),
                                  cute.make_layout((m, k // 2), stride=(k // 2, 1)))
        b_view = cute.make_tensor(cute.recast_ptr(b.iterator, dtype=cutlass.Int32),
                                  cute.make_layout((n, k // 8), stride=(k // 8, 1)))
        sf_view = cute.make_tensor(cute.recast_ptr(sf.iterator, dtype=cutlass.Float8E4M3FN), sf.layout)
        partial_view = None
        if cutlass.const_expr(splits > 1):
            partial_view = cute.make_tensor(partial.iterator,
                                            cute.make_layout((splits, m, n), stride=(m * n, n, 1)))
        gemm(a_view, b_view, sf_view, alpha, output, partial_view, stream)

    a = make_fake_compact_tensor(cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=16)
    b = make_fake_compact_tensor(cutlass.Uint8, (n, k // 2), stride_order=(1, 0), assumed_align=16)
    sf = make_fake_compact_tensor(cutlass.Uint8, (n * k // 16,), assumed_align=16)
    output = make_fake_compact_tensor(cutlass.BFloat16, (m, n), stride_order=(1, 0), assumed_align=16)
    alpha = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    partial = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=16)
    compiled = cute.compile(launch, a, b, sf, output, alpha, partial,
                            make_fake_stream(use_tvm_ffi_env_stream=True), options=f"--gpu-arch {gpu_arch} --enable-tvm-ffi")
    symbol = f"ggml_flashinfer_nvfp4_w4a16_m{m}_n{n}_k{k}_s{splits}_v1"
    object_path = output_dir / f"{symbol}.o"
    compiled.export_to_c(str(object_path), function_name=symbol)
    return object_path


def export_moe(source_dir, output_dir, gpu_arch, sm_counts):
    import importlib.util
    import torch

    if sm_counts is None:
        expected = {"sm_120a": ((12, 0),), "sm_121a": ((12, 1),), "sm_120f": ((12, 0), (12, 1))}
        if torch.cuda.get_device_capability(0) not in expected[gpu_arch]:
            raise ValueError("Automatic MoE export requires a GPU matching --gpu-arch")
        sm_counts = [torch.cuda.get_device_properties(0).multi_processor_count]

    package = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x"
    source_root = source_dir / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x"

    def load_overlay(name, source, patch_name):
        target = output_dir / f"{source.stem}_ggml.py"
        patch = Path(__file__).with_name(patch_name)
        subprocess.run(["patch", "--batch", "--silent", "--fuzz=0", "--output", str(target), str(source), str(patch)], check=True)
        spec = importlib.util.spec_from_file_location(name, target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    gated = load_overlay(package + "._moe_dynamic.gated_ggml", source_root / "_moe_dynamic/gated.py", "flashinfer-moe-kernel.patch")
    dispatch = load_overlay(package + ".moe_dispatch_ggml", source_root / "moe_dispatch.py", "flashinfer-moe-dispatch.patch")

    def make_kernel(**kwargs):
        kwargs.pop("hidden_size")
        kwargs.pop("intermediate_size")
        kwargs["known_topk"] = kwargs.pop("num_topk")
        return gated.MoEGatedDynamicKernel(**kwargs)

    dispatch.MoEDynamicKernel = make_kernel
    dispatch.build_and_load_cute_dsl_kernel = lambda _module, _name, compile_kernel, **kwargs: compile_kernel()
    objects = []
    for sm_count in dict.fromkeys(sm_counts):
        # The selected kernel uses one resident CTA per SM and no multi-CTA clusters.
        dispatch.get_num_sm = lambda _device, count=sm_count: count
        dispatch.get_max_active_clusters = lambda _cluster_size, count=sm_count: count
        compiled, _ = dispatch._get_dynamic_kernel(256, 2048, 2048, 512, 8, 49152,
            fast_math=False, input_scales_are_reciprocal=False, intermediate_size=512)
        object_path = output_dir / f"ggml_flashinfer_nvfp4_moe_sm{sm_count}.o"
        compiled.export_to_c(str(object_path), function_name=f"ggml_flashinfer_nvfp4_moe_sm{sm_count}_v2")
        objects.append(object_path)
    return objects


if __name__ == "__main__":
    main()
