# Flash Ulysses

Fused BF16 QKV projection, Q/K RMSNorm, NeoX RoPE and Ulysses exchange for
single-node Hopper GPUs, built with [ThunderKittens](https://github.com/HazyResearch/ThunderKittens).
Each CTA overlaps WGMMA and peer TMA stores using separate warps on the same SM.

**Known precision issue:** the current FP32 register epilogue does not yet pass
comparisons against the SGLang/PyTorch baselines at `atol=0.01`, `rtol=0.01`.
The performance results below pass only the looser benchmark tolerance check;
the precision issue remains unresolved.

## Build

Requires Linux, CUDA Toolkit 12.8+, CUDA-enabled PyTorch 2.8+, and 2–8 Hopper GPUs
with peer access and CUDA multicast support.

```bash
git submodule update --init --recursive
bash install.sh
```

`install.sh` initializes the pinned ThunderKittens submodule and builds an
editable installation using the existing PyTorch/CUDA environment.

## Use

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

Outputs alias context storage. Before the next invocation, **all physical ranks
must complete the previous invocation and finish using its outputs**. A local
stream dependency alone does not protect a receiver from another GPU reusing
its storage.

## GPU comparisons

One entry point runs both correctness checks and latency comparisons:

```bash
# Independent PyTorch + NCCL baseline; tests tails and buffer reuse by default.
torchrun --standalone --nproc-per-node=8 tests/benchmark.py --sp-size 4

# Both baselines, using an installed SGLang Ulysses checkout.
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
absolute/relative tolerances are `atol=0.03125` and `rtol=0.02`. Every element must
satisfy `abs(actual - expected) <= atol + rtol * abs(expected)` using
`torch.isclose`. Maximum absolute error and RMS error are diagnostic values
reported across ranks; passing is determined by the elementwise tolerance check.

Latency uses eager CUDA events, warmup, no L2 flush, and the maximum rank's mean.
TK timing includes projection, RMSNorm/RoPE, peer stores and its exit barrier;
baseline timing includes projection, normalization, RoPE, packing and NCCL.
Context allocation, parameter preparation and the caller rendezvous are excluded.
Reported TFLOP/s uses
projection FLOPs only. Use `--warmup`, `--iters`, `--hidden-size`, `--q-heads` and
`--kv-heads` to choose the workload.

### H20 results (4 GPUs, FP32 register epilogue)

Measured on 4 NVIDIA H20 GPUs with PyTorch 2.13.0+cu130 and CUDA 13.0.
The workload uses BF16, `H=4096`, Q/KV heads `32/8`, head dimension `128`,
`world=4`, `SP=4` and `base_TP=1`. `L` is the token count **per GPU**; each
output contains `4*L` token rows after exchange.

```bash
torchrun --standalone --nproc-per-node=4 tests/benchmark.py \
  --sp-size 4 --baseline torch sglang \
  --local-tokens 1 63 64 65 127 128 129 1024 2048 4096 8192 \
  --atol 0.03125 --rtol 0.02
```

| Local tokens (L) | TK (ms) | SGLang (ms) | PyTorch (ms) | vs SGLang | vs PyTorch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.111 | 0.273 | 0.478 | 2.46x | 4.30x |
| 63 | 0.114 | 0.263 | 0.495 | 2.30x | 4.32x |
| 64 | 0.114 | 0.312 | 0.570 | 2.74x | 5.00x |
| 65 | 0.131 | 0.334 | 0.622 | 2.55x | 4.76x |
| 127 | 0.116 | 0.263 | 0.498 | 2.26x | 4.29x |
| 128 | 0.116 | 0.260 | 0.498 | 2.24x | 4.30x |
| 129 | 0.187 | 0.265 | 0.500 | 1.42x | 2.67x |
| 1024 | 0.428 | 0.595 | 0.830 | 1.39x | 1.94x |
| 2048 | 0.816 | 1.103 | 1.575 | 1.35x | 1.93x |
| 4096 | 1.590 | 2.117 | 3.017 | 1.33x | 1.90x |
| 8192 | 3.136 | 4.176 | 5.955 | 1.33x | 1.90x |

At the final visit, TK reaches 131.48 GEMM-equivalent TFLOP/s per GPU.

The largest reported errors across the full run, against either baseline, are:

| Output | Maximum absolute error | Maximum RMS error |
| --- | ---: | ---: |
| Q | 0.046875 | 0.003249 |
| K | 0.031250 | 0.003456 |
| V | 0.015625 | 0.000124 |

The optimized path retains FP32 intermediates through Q/K RMSNorm and RoPE,
then rounds to BF16 at the final shared store. The baselines round projection
and RMSNorm outputs to BF16 as separate operators. This changes the numerical
computation and is a factor to investigate; it has not been established as the
sole cause of the stricter-tolerance failures. Precision validation remains
open, including comparison against an FP32-intermediate reference.

## License

Apache-2.0. ThunderKittens retains its own license in the pinned submodule.
