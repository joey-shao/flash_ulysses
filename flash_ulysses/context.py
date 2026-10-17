"""Collective receive-buffer ownership and the public QKV interface."""

import torch
import torch.distributed as dist

from . import _C


class FlashUlyssesContext:
    """Construct, call and close collectively on one Hopper node.

    rank and world_size describe the participating physical GPUs on this node.
    control_group must cover those same participants for lifecycle barriers.
    Calls use the current CUDA stream. Results alias Q/K/V until the next call
    or close. Before reuse, every physical rank must finish consuming its
    previous output and complete the previous TK barrier. See DESIGN.md.
    """

    def __init__(
        self,
        *,
        rank,
        world_size,
        max_local_tokens,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        sp_size=2,
        control_group=None,
    ):
        self.rank, self.world_size = rank, world_size
        self.control_group = control_group
        self.hidden_size = hidden_size
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim, self.sp_size = head_dim, sp_size
        self.max_local_tokens = max_local_tokens
        self.device = torch.device("cuda", torch.cuda.current_device())

        if not 2 <= sp_size <= self.world_size <= 8 or self.world_size % sp_size:
            raise ValueError("SP must divide a single-node world of 2 to 8 GPUs")

        if hidden_size <= 0 or hidden_size % 64 or head_dim != 128:
            raise ValueError(
                "Requires hidden_size > 0 divisible by 64 and head_dim=128"
            )

        if (
            num_heads <= 0
            or num_kv_heads <= 0
            or num_heads % num_kv_heads
            or num_heads % self.world_size
            or num_kv_heads % self.world_size
        ):
            raise ValueError(
                "Q heads must divide by KV heads; both must divide by world_size"
            )

        if torch.cuda.get_device_capability() != (9, 0):
            raise ValueError("Requires SM90 (H100/H200)")

        capacity = sp_size * max_local_tokens
        self.Q = _C.TKParallelTensor(
            (capacity, self.output_q_heads * head_dim),
            torch.bfloat16,
            self.rank,
            self.world_size,
            False,
        )
        self.K = _C.TKParallelTensor(
            (capacity, self.output_kv_heads * head_dim),
            torch.bfloat16,
            self.rank,
            self.world_size,
            False,
        )
        self.V = _C.TKParallelTensor(
            (capacity, self.output_kv_heads * head_dim),
            torch.bfloat16,
            self.rank,
            self.world_size,
            False,
        )
        self.barrier = _C.TKParallelTensor(
            (1, 1),
            torch.int,
            self.rank,
            self.world_size,
            True,
        )
        self.barrier.data_.zero_()
        torch.cuda.current_stream().synchronize()
        dist.barrier(group=control_group)

    @property
    def base_tp_size(self):
        return self.world_size // self.sp_size

    @property
    def local_q_heads(self):
        return self.num_heads // self.base_tp_size

    @property
    def local_kv_heads(self):
        return self.num_kv_heads // self.base_tp_size

    @property
    def output_q_heads(self):
        return self.num_heads // self.world_size

    @property
    def output_kv_heads(self):
        return self.num_kv_heads // self.world_size

    @property
    def packed_width(self):
        return (self.local_q_heads + 2 * self.local_kv_heads) * self.head_dim

    @torch.no_grad()
    def ulysses_qkv_forward(
        self,
        hidden_states,
        qkv_weight,
        positions,
        q_norm_weight,
        k_norm_weight,
        rope_cache,
        *,
        q_epsilon=1e-6,
        k_epsilon=1e-6,
    ):
        """Return BF16 (q, k, v), each with L * sp_size rows.

        Inputs are contiguous CUDA tensors on the current/context device:
        hidden_states BF16 [L, H], qkv_weight BF16 [H, packed_width],
        positions int64 [L], norm weights FP32 [128], and a NeoX RoPE cache
        FP32 [max_position, 128] containing [cos(64), sin(64)]. Positions must
        index the cache. Epsilons must be positive. The caller crops padding.
        """
        local_tokens = hidden_states.size(0)
        if not 0 < local_tokens <= self.max_local_tokens:
            raise ValueError("Input length must be within [1, max_local_tokens]")

        _C.launch(
            hidden_states,
            qkv_weight,
            positions,
            q_norm_weight,
            k_norm_weight,
            rope_cache,
            self.Q,
            self.K,
            self.V,
            self.barrier,
            self.local_q_heads,
            self.local_kv_heads,
            self.sp_size,
            q_epsilon,
            k_epsilon,
        )

        output_tokens = local_tokens * self.sp_size
        return tuple(t.data()[:output_tokens] for t in (self.Q, self.K, self.V))

    def close(self):
        """Collectively release peer mappings after all output consumers finish."""
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.control_group)
        # Native TK destructors own cleanup; do not call destroy() a second time.
        with torch.cuda.device(self.device):
            self.Q = None
            self.K = None
            self.V = None
            self.barrier = None
