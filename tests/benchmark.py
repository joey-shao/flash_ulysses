"""torchrun --standalone --nproc-per-node=8 tests/benchmark.py --sp-size 4"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU QKV correctness and latency comparisons"
    )
    parser.add_argument(
        "--baseline", nargs="+", choices=("torch", "sglang"), default=["torch"]
    )
    parser.add_argument("--sp-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument(
        "--local-tokens",
        nargs="+",
        type=int,
        default=[1, 63, 64, 65, 127, 128, 129, 1024, 4096],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--atol", type=float, default=0.03125)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if min(args.local_tokens) < 1 or args.warmup < 0 or args.iters < 1:
        parser.error("local-tokens/iters must be positive; warmup must be nonnegative")
    return args


def run(context, parameters, references, control_group, local_tokens, args, seed):
    import torch
    import torch.distributed as dist

    from common import benchmark_no_l2_clear, check_diff, clean_print, synchronize

    generator = torch.Generator(device="cuda").manual_seed(
        seed + context.rank % context.sp_size
    )
    x = torch.randn(
        local_tokens,
        context.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    tokens = (
        torch.arange(local_tokens, device="cuda")
        + context.rank % context.sp_size * local_tokens
    )
    positions = (tokens * 7 + 19) % parameters["rope_cache"].size(0)
    # Exercise user padding and an SP-empty source when L=1.
    real_tokens = local_tokens * context.sp_size - 1
    x[tokens >= real_tokens] = 0
    positions[tokens >= real_tokens] = 0
    tk_run = lambda: context.ulysses_qkv_forward(x, positions=positions, **parameters)
    calls = {
        name: (lambda ref=ref: ref(x, positions)) for name, ref in references.items()
    }
    clean_print(
        f"\nL={local_tokens} H={context.hidden_size} Q/KV={context.num_heads}/{context.num_kv_heads} "
        f"world={context.world_size} SP={context.sp_size} base_TP={context.base_tp_size}"
    )

    for tensor in (context.Q, context.K, context.V):
        tensor.data_.fill_(float("nan"))
    synchronize(control_group)
    actual = tk_run()
    for name, fn in calls.items():
        expected = fn()
        for component, got, want in zip("QKV", actual, expected):
            check_diff(
                f"TK/{name} {component}", got, want, atol=args.atol, rtol=args.rtol
            )

    # A short active shape must not overwrite unused receive capacity.
    active = local_tokens * context.sp_size
    untouched = (
        torch.stack(
            [
                torch.isnan(t.data_[active:]).all()
                for t in (context.Q, context.K, context.V)
            ]
        )
        .all()
        .int()
    )
    dist.all_reduce(untouched, op=dist.ReduceOp.MIN)
    assert untouched.item(), "Kernel wrote beyond the active output"
    clean_print("  correctness: PASS (all Q/K/V rows and output boundary)")
    if args.check_only:
        return

    timings = {
        name: benchmark_no_l2_clear(fn, control_group, args.warmup, args.iters)
        for name, fn in {"TK": tk_run, **calls}.items()
    }
    flops = 2 * local_tokens * context.hidden_size * context.packed_width
    for name, ms in timings.items():
        clean_print(
            f"  {name:7s}: {ms:.3f} ms | GEMM-equivalent {flops / (ms * 1e9):.2f} TFLOP/s"
        )
    for name in calls:
        clean_print(f"  TK vs {name}: {timings[name] / timings['TK']:.2f}x")


def main(args):
    import torch

    from common import (
        clean_print,
        destroy_distributed_environment,
        init_distributed_environment,
    )
    from flash_ulysses import FlashUlyssesContext
    from torch_base import TorchReference, make_parameters

    rank, world_size, control_group = init_distributed_environment()

    context = FlashUlyssesContext(
        rank=rank,
        world_size=world_size,
        max_local_tokens=max(args.local_tokens) + 128,
        control_group=control_group,
        hidden_size=args.hidden_size,
        num_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        sp_size=args.sp_size,
    )
    parameters = make_parameters(context)
    references = {}
    if "sglang" in args.baseline:
        from sglang_base import SGLangReference

        references["SGLang"] = SGLangReference(context, parameters)
    if "torch" in args.baseline:
        references["PyTorch"] = TorchReference(context, parameters)

    clean_print(
        f"{torch.cuda.get_device_name()} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}"
    )
    clean_print(
        "Eager CUDA events; max rank mean; no L2 flush; caller rendezvous excluded."
    )
    with torch.inference_mode():
        # Large -> small -> tails -> large reuses one context with changing inputs.
        cases = [max(args.local_tokens), *args.local_tokens]
        for i, local_tokens in enumerate(cases):
            run(
                context,
                parameters,
                references,
                control_group,
                local_tokens,
                args,
                seed=10000 + i * 101,
            )
    context.close()
    for reference in references.values():
        reference.close()
    destroy_distributed_environment(control_group)


if __name__ == "__main__":
    main(parse_args())
