"""Shared utilities for the GPU correctness and latency comparisons."""

import os

import torch
import torch.distributed as dist


def init_distributed_environment():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    control_group = dist.new_group(backend="gloo")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return local_rank, dist.get_world_size(), control_group


def destroy_distributed_environment(control_group):
    dist.destroy_process_group(control_group)
    dist.destroy_process_group()


def clean_print(message):
    if dist.get_rank() == 0:
        print(message, flush=True)


def synchronize(control_group):
    # Every receiver must finish before a peer overwrites its output or reuses
    # TK's scalar barrier. This rendezvous is outside the timed region.
    torch.cuda.synchronize()
    dist.barrier(group=control_group)


def check_diff(name, actual, expected, *, atol, rtol):
    diff = (actual.float() - expected.float()).abs()
    stats = torch.stack((diff.max(), diff.square().mean().sqrt()))
    passed = torch.isclose(actual, expected, atol=atol, rtol=rtol).all().int()
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    clean_print(f"  {name}: max_abs={stats[0].item():.6f}, rms={stats[1].item():.6f}")
    assert passed.item(), f"{name} differs from baseline (atol={atol}, rtol={rtol})"


def benchmark_no_l2_clear(fn, control_group, num_warmup_iters=3, num_iters=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for i in range(num_warmup_iters + num_iters):
        synchronize(control_group)
        start.record()
        fn()
        end.record()
        end.synchronize()
        if i >= num_warmup_iters:
            elapsed.append(start.elapsed_time(end))
    latency = torch.tensor(sum(elapsed) / len(elapsed), device="cuda")
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return latency.item()
