"""Check the exported SM120 MoE kernel against sparse, independently scaled experts.

Run with the export environment: python tests/test-flashinfer-moe.py --library LIBRARY
"""

import argparse

import torch
import tvm_ffi

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--library", required=True)
parser.add_argument("--tokens", type=int, choices=(128, 129, 512, 2048), default=128)
args = parser.parse_args()

E, H, I, T = 256, 2048, 512, args.tokens
D = "cuda"
torch.manual_seed(8)
up = torch.zeros((E, I, H // 2), dtype=torch.uint8, device=D)
gate = up.clone()
down = torch.zeros((E, H, I // 2), dtype=torch.uint8, device=D)
for e in range(E):
    idx = (torch.arange(I, device=D) + e) % H
    up[e, torch.arange(I, device=D), idx // 2] = (2 << ((idx % 2) * 4)).to(torch.uint8)
    gate[e, torch.arange(I, device=D), idx // 2] = (4 << ((idx % 2) * 4)).to(torch.uint8)
    idx = torch.arange(H, device=D) % I
    down[e, torch.arange(H, device=D), idx // 2] = (2 << ((idx % 2) * 4)).to(torch.uint8)
usf = torch.full((E * I * H // 16,), 0x38, dtype=torch.uint8, device=D)
gsf = usf.clone()
dsf = torch.full((E * H * I // 16,), 0x38, dtype=torch.uint8, device=D)
x = (torch.randint(-4, 5, (T, H), device=D) * 0.25).bfloat16()
weights = torch.softmax(torch.randn((T, 8), device=D), -1).contiguous()
is_ = torch.ones(E, device=D) * 0.01
ds = torch.ones(E, device=D) * 0.001
alpha = torch.stack((torch.linspace(0.5, 1, E, device=D), torch.linspace(0.75, 1.25, E, device=D)), 1) * is_[:, None]
da = ds.clone()
mod = tvm_ffi.load_module(args.library)
f = mod[f"ggml_flashinfer_nvfp4_moe_sm{torch.cuda.get_device_properties(0).multi_processor_count}_v2"]
vals = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=D)


def quant(x, gs):
    shape = x.shape
    z = x.reshape(-1, 16)
    gs = gs.repeat_interleave(z.shape[0] // gs.numel())[:, None]
    sf = (z.abs().amax(1, keepdim=True) / (6 * gs)).clamp(max=448).to(torch.float8_e4m3fn).float()
    scale = sf * gs
    v = z / scale.clamp(min=1e-30)
    q = vals[(v.abs()[..., None] - vals).abs().argmin(-1)] * v.sign()
    return (q * scale).reshape(shape)


for mode in ["balanced", "skewed", "concentrated"]:
    ids = (
        (
            (torch.arange(T * 8, device=D).reshape(T, 8) % E)
            if mode == "balanced"
            else torch.arange(8, device=D).expand(T, 8)
        )
        .int()
        .contiguous()
    )
    if mode == "skewed":
        ids[T // 2 :] = (torch.arange((T - T // 2) * 8, device=D).reshape(-1, 8) % E).int()
    tiles = (T * 8 + 127) // 128 + E - 1
    tasks = 2 * tiles
    rows = tiles * 128
    packed = torch.empty((rows * H // 2,), device=D, dtype=torch.uint8)
    sf = torch.empty((rows * H // 16,), device=D, dtype=torch.uint8)
    state = torch.zeros(6 + 3 * E + 2 * tasks + rows, device=D, dtype=torch.int32)
    tw = torch.empty(rows, device=D)
    out = torch.empty_like(x)

    def addr(x):
        return x.data_ptr()

    call_args = (
        addr(x),
        addr(ids),
        addr(weights),
        addr(packed),
        addr(sf),
        addr(packed),
        addr(sf),
        *[state[i : i + 1] for i in range(5)],
        addr(state[6 + 3 * E :]),
        addr(state[6 + 3 * E + tasks :]),
        up.permute(1, 2, 0).view(torch.float4_e2m1fn_x2),
        addr(usf),
        gate.permute(1, 2, 0).view(torch.float4_e2m1fn_x2),
        addr(gsf),
        down.permute(1, 2, 0).view(torch.float4_e2m1fn_x2),
        addr(dsf),
        state[5 : 5 + E],
        state[5 + E : 5 + 2 * E],
        state[5 + 2 * E : 6 + 3 * E],
        is_,
        alpha.flatten(),
        da,
        ds,
        addr(out),
        addr(state[6 + 3 * E + 2 * tasks :]),
        addr(tw),
        T,
        rows,
        rows,
        tasks,
    )
    f(*call_args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with tvm_ffi.use_torch_stream(torch.cuda.graph(graph)):
        f(*call_args)
    for _ in range(2):
        out.fill_(float("nan"))
        graph.replay()
    xx = quant(x.float()[:, None, :].expand(-1, 8, -1).contiguous(), is_[ids.flatten()])
    idx = (torch.arange(I, device=D)[None, None, :] + ids[:, :, None]) % H
    u = xx.gather(2, idx) * ((alpha[:, 0] / is_)[ids])[:, :, None]
    g = xx.gather(2, idx) * 2 * ((alpha[:, 1] / is_)[ids])[:, :, None]
    act = (torch.nn.functional.silu(g) * u).bfloat16().float()
    act = quant(act, ds[ids.flatten()])
    ref = (act.repeat(1, 1, 4) * weights[:, :, None]).sum(1)
    torch.cuda.synchronize()
    diff = out.float() - ref
    nmse = float(diff.square().sum() / ref.square().sum())
    assert torch.isfinite(out).all() and nmse < 1e-4, (mode, nmse)
    print(f"{mode}: NMSE={nmse:.8f}", flush=True)
