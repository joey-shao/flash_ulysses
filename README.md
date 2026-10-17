# Flash Ulysses

Fused BF16 QKV projection, Q/K RMSNorm, NeoX RoPE and Ulysses exchange for
single-node Hopper GPUs, built with [ThunderKittens](https://github.com/HazyResearch/ThunderKittens).
Each CTA overlaps WGMMA and peer TMA stores using separate warps on the same SM.

## Build

Requires Linux, CUDA Toolkit 12.8+, CUDA-enabled PyTorch 2.8+, and 2–8 Hopper GPUs
with peer access and CUDA multicast support. All participating GPUs must be
visible to every worker; launch one process per GPU with `torchrun`.

```bash
git submodule update --init --recursive
bash install.sh
```

`install.sh` initializes the pinned ThunderKittens submodule and builds an
editable installation using the existing PyTorch/CUDA environment.

## Use

Initialize `torch.distributed` and select `LOCAL_RANK` with `torch.cuda.set_device`
before constructing the context. Construction, forward and close are collective.
All ranks use the same dimensions and active input length.

```python
from flash_ulysses import FlashUlyssesContext

context = FlashUlyssesContext(
    rank=rank,
    world_size=world_size,
    max_local_tokens=4096,
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,
    sp_size=4,
)
q, k, v = context.ulysses_qkv_forward(
    hidden_states, qkv_weight, positions,
    q_norm_weight, k_norm_weight, rope_cache,
    q_epsilon=1e-6, k_epsilon=1e-6,
)
# Cropping belongs to the caller; the operator returns every input row.
q, k, v = (tensor[:real_tokens] for tensor in (q, k, v))
# Consume the outputs before collective close or reuse.
context.close()
```

Let `P` be the physical GPU count, `S` the SP size, and `T=P/S` the base TP size.
`num_heads=Q` and `num_kv_heads=KV` are the model's total head counts. SP groups
contain consecutive ranks. Both head counts must be divisible by `P`, and `Q`
must be divisible by `KV`. `S >= 2`; KV replication is unsupported.

| Input | Dtype | Shape |
| --- | --- | --- |
| `hidden_states` | BF16 | `[L, H]`, `1 <= L <= max_local_tokens`, `H % 64 == 0` |
| `qkv_weight` | BF16 | `[H, (Q + 2*KV)/T * 128]`, packed Q then K then V |
| `positions` | int64 | `[L]`, valid indices into `rope_cache` |
| `q_norm_weight`, `k_norm_weight` | FP32 | `[128]` each |
| `rope_cache` | FP32 | `[max_position, 128]`, packed `cos(64), sin(64)` |

Inputs must be contiguous and on the context's current CUDA device. Epsilons
must be positive. The API is inference-only and launches on the current stream.
Outputs have shapes `[L*S, Q/P*128]`, `[L*S, KV/P*128]`, `[L*S, KV/P*128]`.
`L` need not be a multiple of 128.

Outputs alias context storage. Before the next invocation, **all physical ranks
must complete the previous invocation and finish using its outputs**. A local
stream dependency alone does not protect a receiver from another GPU reusing
its storage. The test harness supplies this rendezvous outside timing. See
[DESIGN.md](DESIGN.md) for the barrier and buffer lifetime contract.

## GPU comparisons

One entry point runs both correctness checks and latency comparisons:

```bash
# Independent PyTorch + NCCL baseline; tests tails and buffer reuse by default.
torchrun --standalone --nproc-per-node=8 tests/benchmark.py --sp-size 4

# Both baselines, using an installed SGLang Ulysses checkout.
PYTHONPATH=/path/to/sglang/python \
  torchrun --standalone --nproc-per-node=8 tests/benchmark.py \
  --sp-size 4 --baseline torch sglang

# Fast correctness-only sweep, including L=129.
torchrun --standalone --nproc-per-node=8 tests/benchmark.py \
  --sp-size 4 --check-only --local-tokens 1 63 64 65 127 128 129
```

The SGLang base follows Qwen3's eager CUDA path directly: BF16 `F.linear`
(the unquantized torch GEMM backend), Q/K/V split, `fused_inplace_qknorm`,
NeoX `apply_rope_with_cos_sin_cache_inplace`, then QKV exchange.

All implementations use the same synthetic weights and FP32 RoPE cache, with
BF16 norm weights exported to TK as FP32. Checks compare every Q/K/V row, including
padding, and verify that inactive receive capacity stays untouched. The default
absolute/relative tolerances are `0.03125`/`0.02`; maximum absolute error and RMS
error are reported across ranks.

Latency uses eager CUDA events, warmup, no L2 flush, and the maximum rank's mean.
TK timing includes projection, epilogue, peer stores and its exit barrier;
baseline timing includes projection, normalization, RoPE, packing and NCCL.
Context allocation, parameter preparation and the caller rendezvous are excluded.
Reported TFLOP/s uses
projection FLOPs only. Use `--warmup`, `--iters`, `--hidden-size`, `--q-heads` and
`--kv-heads` to choose the workload. There are no CPU tests.

## License

Apache-2.0. ThunderKittens retains its own license in the pinned submodule.
