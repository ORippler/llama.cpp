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
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.source_dir), "rev-parse", "HEAD"], text=True).strip()
    if revision != FLASHINFER_COMMIT:
        parser.error(f"FlashInfer must be checked out at {FLASHINFER_COMMIT}, found {revision}")
    if subprocess.run(["git", "-C", str(args.source_dir), "diff", "--quiet", "HEAD", "--", "flashinfer"]).returncode:
        parser.error("FlashInfer sources must match the pinned checkout")
    sys.path.insert(0, str(args.source_dir.resolve()))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(args.output_dir.resolve()))
    os.environ["CUTE_DSL_ARCH"] = args.gpu_arch

    import cutlass.cute as cute  # ty: ignore[unresolved-import]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects = [export_attention(args.output_dir, q_heads, kv_heads, args.gpu_arch)
               for q_heads, kv_heads in ((24, 4), (16, 2))]

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


if __name__ == "__main__":
    main()
