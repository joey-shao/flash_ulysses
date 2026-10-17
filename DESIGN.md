# Kernel design

Flash Ulysses fuses `X @ W_qkv`, headwise Q/K RMSNorm, NeoX RoPE and SP exchange.
The implementation lives in
[`flash_ulysses_kernel_hopper.cu`](flash_ulysses/csrc/impl/flash_ulysses_kernel_hopper.cu).
It uses upstream ThunderKittens `gl`, `pgl`, WGMMA, TMA, semaphores and launch APIs.

## Work and rank layout

A task computes **128 token rows × one 128-element head**. The reduction advances
by 64 hidden elements. The persistent grid uses
`min(num_tasks, device_SM_count)` CTAs.

```text
head_count = q_heads + 2 * kv_heads       # heads in this base-TP projection
row_idx    = task_id / head_count
col_idx    = task_id % head_count        # packed Q, then K, then V
```

There are `ceil(L/128) * head_count` tasks, without a task table. Each GPU first
projects all heads in its base TP shard. Let physical world size be `P`, SP size
be `S`, and base TP size be `T=P/S`. Physical rank is `base_tp_rank*S + sp_rank`.
Each SP group contains `S` consecutive ranks. Base TP groups connect equal SP
ranks across those groups.

After selecting Q, K or V from `col_idx`, subtract its component offset to obtain
`head_id`. For that component, `heads_per_gpu = model_heads/P`:

```text
dst_gpu = group_start + head_id / heads_per_gpu
dst_col = head_id % heads_per_gpu
```

For example, `P=8, S=4, Q=32, KV=8`: base TP size is 2; each projection produces
16 Q heads and 4 K/V heads. GPU 2 belongs to SP group `[0,1,2,3]` and has source
SP rank 2. Its Q head 9 goes to GPU 2, head column 1. Its K head 3 goes to GPU 3,
head column 0. Every destination receives all four sources' tokens.

Each output element has one writer. Peer TMA uses **store**, without reduction
or atomic add. Base TP selects projection columns; it does not split the GEMM
reduction dimension.

## CTA pipeline

Every CTA has 384 threads, with the same roles on every SM:

```text
WG0 / WG1: WGMMA → BF16 shared output → Q/K RMSNorm + RoPE
WG2:
  warp0, lane0: TMA load X and QKV weight
  warp1, lane0: TMA store Q/K/V directly to the destination GPU
  warp2 / warp3: inactive
```

Two consumers compute 64 rows each and share the weight tile. Three input slots
hold two `[64,64]` activation tiles and one `[64,128]` weight tile each. Two
output slots each hold two `[64,128]` tiles. These rings use 160 KiB of tile
storage, plus semaphore storage and alignment. They do not overlap in memory.
Consumers request 192 registers per thread; producers release down to 40.
Actual register use/spills must be checked in the `ptxas -v` build output.

| Semaphore | Signals completion of | Arrival count |
| --- | --- | --- |
| `inputs_arrived[3]` | TMA input loads | One producer arrival + expected bytes |
| `inputs_finished[3]` | WGMMA reading the slot | Eight consumer warps |
| `outputs_arrived[2]` | Both consumer epilogues | Two consumer warpgroups |
| `outputs_finished[2]` | Peer TMA stores | One store thread |

One per-thread `phasebits` tracks all rings: lower bits for arrived semaphores,
upper bits for finished semaphores. Finished phases start at 1 so initially empty
slots are available. `input_stage` and `output_stage` advance independently and
carry across tasks.

`__syncthreads()` publishes semaphore initialization before roles diverge.
A named barrier within each consumer WG completes shared output stores before
its epilogue. Every consumer executes the generic-to-async proxy fence; an
eight-warp barrier joins both consumers before handing the slot to the store
warp. A full `tma::store_async_wait()` completes peer writes before releasing the
slot. These are producer/consumer dependencies, not optional input validation.

## Numerics and token tails

WGMMA accumulates in FP32 and rounds the projection to BF16. Q and K then use
FP32 RMSNorm, round its result to BF16, apply FP32 NeoX RoPE, and round to BF16
again. V keeps the BF16 projection. The intermediate rounding boundaries match
the separate operators in the reference path. Different reduction orders can
still produce small differences, so comparisons use explicit tolerances.

Q/K epilogues load and store rows with TK `warp::load` / `warp::store`. Each warp
owns rows `warp_id, warp_id+4, ...` in its 64-row tile. Rows past `L` skip the
epilogue, including the position lookup. Out-of-bounds TMA activation loads fill
with zero.

Each peer output is described as **`[S, 1, L, heads_per_gpu*128]`** using the native
`parallel_tensor_to_pgl` shape overload. The underlying allocation is a contiguous
2D capacity buffer. TMA coordinates are:

```text
{source_sp_rank, 0, row_idx*2 + consumer_wg, dst_col}
```

The source rank is a batch coordinate. TMA clips tail stores at that batch's `L`
rows, so a partial tile cannot spill into the next source's token slice. No
pointer offset, `source_slice`, staging buffer or row padding is required.
For `L=129, S=4`, outputs contain 516 rows, with each source occupying exactly
129 rows. The caller handles application padding and real-token cropping.

## Completion and reuse

The compute kernel finishes all peer stores. A second, one-thread kernel then
calls TK `barrier_all` on a multicast `TKParallelTensor((1,1), torch.int, ...)`.
Stream order joins all local CTAs before this barrier joins the physical world.
The subsequent system fence orders consumption after TK's relaxed polling load.
Downstream work on the same stream can consume the received tensors.

This scalar TK barrier is **not safe for overlapping generations**: a fast GPU
can enter again before another GPU observes/reset its previous counter. Also,
a fast producer can overwrite a slow receiver's output. Therefore the caller
must order the next invocation after all physical ranks have completed the
previous barrier and consumed their outputs. Local default-stream ordering
alone does not establish that cross-GPU dependency. This includes ranks outside
the immediate SP group when base TP size is greater than one.

The GPU harness uses CUDA synchronization followed by a world Gloo rendezvous
before each call; it excludes this rendezvous from latency. Production callers
can use an existing equivalent dependency. There is no begin flag, epoch,
timeout or internal entry barrier. Arbitrary back-to-back CUDA graph replay is
not covered by this contract.

The context owns three native TK peer allocations and one multicast barrier.
Inputs are recorded on the launch stream. Output views alias receive storage;
all consumers must finish before reuse or collective `close()`. The caller must
also order input readiness when switching streams. Upstream TK uses fixed IPC
broker names, so concurrent jobs require separate IPC/socket namespaces.

## References

- [TK H100 GEMM-RS](thirdparty/ThunderKittens/kernels/parallel/gemm_rs/gemm_rs_h100.cu): CTA roles and native launch pattern.
- [TK synchronization](thirdparty/ThunderKittens/include/ops/thread/util/sync.cuh): semaphore and multicast barrier implementation.
- [CUDA asynchronous copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html): TMA bounds, proxy ordering and store completion.
