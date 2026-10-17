# SPDX-License-Identifier: Apache-2.0
"""Qwen3 CUDA operators and a local copy of the Ulysses QKV exchange.

Source: the local Projects/sglang Ulysses checkout:
  srt/models/qwen3.py: Qwen3Attention.forward_prepare_native
  srt/models/utils.py: apply_qk_norm (eager CUDA fused path)
  srt/layers/rotary_embedding/base.py: RotaryEmbedding.forward_cuda
  srt/layers/sp/sp_strategy.py: exchange_qkv packing and unpacking
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F


class SGLangReference:
    def __init__(self, context, parameters):
        from sglang.kernels.ops.attention.rope import (
            apply_rope_with_cos_sin_cache_inplace,
        )
        from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

        self.context, self.parameters = context, parameters
        self.qk_norm = fused_inplace_qknorm
        self.rope = apply_rope_with_cos_sin_cache_inplace

        # UnquantizedLinearMethod's torch backend uses F.linear with [N, H]
        # contiguous weights. Prepare that layout outside the timed region.
        self.weight = parameters["qkv_weight"].t().contiguous()
        # Qwen3's ordinary BF16 path uses BF16 norm weights; TK takes FP32
        # copies of those same values. The fused kernel requires input dtype.
        self.q_norm_weight = parameters["q_norm_weight"].to(torch.bfloat16)
        self.k_norm_weight = parameters["k_norm_weight"].to(torch.bfloat16)
        for start in range(0, context.world_size, context.sp_size):
            ranks = list(range(start, start + context.sp_size))
            group = dist.new_group(ranks, backend="nccl")
            if context.rank in ranks:
                self.group = group

    @torch.no_grad()
    def __call__(self, x, positions):
        c, p = self.context, self.parameters
        widths = (
            c.local_q_heads * c.head_dim,
            c.local_kv_heads * c.head_dim,
            c.local_kv_heads * c.head_dim,
        )
        qkv = F.linear(x, self.weight)
        q, k, v = qkv.split(widths, dim=-1)
        q_by_head = q.view(x.size(0), -1, c.head_dim)
        k_by_head = k.view(x.size(0), -1, c.head_dim)
        self.qk_norm(
            q=q_by_head,
            k=k_by_head,
            q_weight=self.q_norm_weight,
            k_weight=self.k_norm_weight,
            head_dim=c.head_dim,
            eps=p["q_epsilon"],
        )
        self.rope(
            positions=positions,
            q=q_by_head,
            k=k_by_head,
            cos_sin_cache=p["rope_cache"],
            is_neox=True,
        )
        # Norm and RoPE mutate the Q/K views of qkv; V remains unmodified.
        # Keep all L*SP rows, including caller padding, to match TK's contract.
        parts = [t.reshape(x.size(0), c.sp_size, -1) for t in (q, k, v)]
        send = torch.cat(parts, dim=-1).transpose(0, 1).contiguous()
        received = torch.empty_like(send)
        dist.all_to_all_single(received.view(-1), send.view(-1), group=self.group)
        output = received.reshape(x.size(0) * c.sp_size, -1)
        return tuple(
            t.contiguous()
            for t in output.split([w // c.sp_size for w in widths], dim=-1)
        )

    def close(self):
        dist.destroy_process_group(self.group)
