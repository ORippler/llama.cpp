import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


FLASHINFER_COMMIT = "8ec9464ce81d81c9e3670aa09acd1a8780f83695"


def main():
    parser = argparse.ArgumentParser(description="Export SM120 FlashInfer prefill kernels to a TVM FFI library")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cc", default="cc", help="Host linker driver")
    parser.add_argument("--kernels", choices=("attention", "gemm", "moe", "all"), default="attention")
    parser.add_argument("--max-active-clusters", type=int, default=128)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.source_dir), "rev-parse", "HEAD"], text=True).strip()
    if revision != FLASHINFER_COMMIT:
        parser.error(f"FlashInfer must be checked out at {FLASHINFER_COMMIT}, found {revision}")
    if subprocess.run(["git", "-C", str(args.source_dir), "diff", "--quiet", "HEAD", "--", "flashinfer"]).returncode:
        parser.error("FlashInfer sources must match the pinned checkout")
    if args.max_active_clusters <= 0:
        parser.error("--max-active-clusters must be positive")
    sys.path.insert(0, str(args.source_dir.resolve()))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(args.output_dir.resolve()))

    import cutlass.cute as cute  # ty: ignore[unresolved-import]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects = []
    if args.kernels in ("attention", "all"):
        for q_heads, kv_heads in ((24, 4), (16, 2)):
            objects.append(export_attention(args.output_dir, q_heads, kv_heads))
    if args.kernels in ("gemm", "all"):
        objects.append(export_gemm(args.output_dir, args.max_active_clusters))
    if args.kernels in ("moe", "all"):
        objects.append(export_moe(args.source_dir, args.output_dir))

    libraries = [Path(lib) for lib in cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)]
    for library in libraries:
        target = args.output_dir / library.name
        if library.resolve() != target.resolve():
            shutil.copy2(library, target)
    subprocess.run([args.cc, "-shared", "-o", str(args.output_dir / "libggml_flashinfer_fp8.so"),
                    *map(str, objects), f"-L{args.output_dir}", "-Wl,-rpath,$ORIGIN",
                    *[f"-l:{lib.name}" for lib in libraries]], check=True)


def export_attention(output_dir, q_heads, kv_heads):
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
                            make_fake_stream(use_tvm_ffi_env_stream=True), options="--gpu-arch sm_120a --enable-tvm-ffi")
    symbol = "ggml_flashinfer_fp8_v1" if (q_heads, kv_heads) == (24, 4) else "ggml_flashinfer_fp8_h16_kv2_v1"
    object_path = output_dir / f"{symbol}.o"
    compiled.export_to_c(str(object_path), function_name=symbol)
    return object_path


def export_gemm(output_dir, max_active_clusters):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    from cuda.bindings import driver as cuda
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import DenseGemmKernel

    gemm = DenseGemmKernel(16, (128, 128), (1, 1), enable_pdl=False)

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
                            options="--gpu-arch sm_120a --enable-tvm-ffi")
    object_path = output_dir / "ggml_flashinfer_nvfp4_128.o"
    compiled.export_to_c(str(object_path), function_name="ggml_flashinfer_nvfp4_gemm_v1")
    return object_path


def export_moe(source_dir, output_dir):
    import importlib.util
    import torch

    package = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x"
    source_root = source_dir / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x"

    def load_overlay(name, source, patch_name):
        target = output_dir / f"{source.stem}_ggml.py"
        patch = Path(__file__).with_name(patch_name)
        subprocess.run(["patch", "--batch", "--silent", "--output", str(target), str(source), str(patch)], check=True)
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
    torch.cuda.synchronize()
    compiled, _ = dispatch._get_dynamic_kernel(256, 2048, 2048, 512, 8, 49152,
        fast_math=False, input_scales_are_reciprocal=False, intermediate_size=512)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    object_path = output_dir / "ggml_flashinfer_nvfp4_moe.o"
    compiled.export_to_c(str(object_path), function_name=f"ggml_flashinfer_nvfp4_moe_sm{sm_count}_v2")
    return object_path


if __name__ == "__main__":
    main()
