"""Eager PyTorch projection, headwise RMSNorm/NeoX RoPE, and NCCL exchange."""

import torch
import torch.distributed as dist


def make_parameters(context, seed=1234):
    generator = torch.Generator(device="cuda").manual_seed(
        seed + context.rank // context.sp_size
    )
    weight = (
        torch.randn(
            context.hidden_size,
            context.packed_width,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * context.hidden_size**-0.5
    )
    q_norm = torch.empty(128, device="cuda", dtype=torch.bfloat16)
    k_norm = torch.empty_like(q_norm)
    q_norm.uniform_(0.5, 1.5, generator=generator)
    k_norm.uniform_(0.5, 1.5, generator=generator)
    inv_freq = 1.0 / (
        1_000_000 ** (torch.arange(0, 128, 2, device="cuda").float() / 128)
    )
    angles = torch.arange(40960, device="cuda").float()[:, None] * inv_freq
    return dict(
        qkv_weight=weight,
        q_norm_weight=q_norm.float(),
        k_norm_weight=k_norm.float(),
        rope_cache=torch.cat((angles.cos(), angles.sin()), dim=-1),
        q_epsilon=1e-6,
        k_epsilon=1e-6,
    )


class TorchReference:
    def __init__(self, context, parameters):
        self.context, self.parameters = context, parameters
        for start in range(0, context.world_size, context.sp_size):
            ranks = list(range(start, start + context.sp_size))
            group = dist.new_group(ranks, backend="nccl")
            if context.rank in ranks:
                self.group = group

    @torch.no_grad()
    def __call__(self, x, positions):
        c, p = self.context, self.parameters
        widths = (c.local_q_heads * 128, c.local_kv_heads * 128, c.local_kv_heads * 128)
        q, k, v = (x @ p["qkv_weight"]).split(widths, dim=-1)
        cos, sin = p["rope_cache"][positions].unsqueeze(1).chunk(2, dim=-1)

        def norm_rope(tensor, gamma, epsilon):
            value = tensor.reshape(x.size(0), -1, 128).float()
            value = value * torch.rsqrt(
                value.square().mean(dim=-1, keepdim=True) + epsilon
            )
            # Match the BF16 boundary between RMSNorm and RoPE.
            value = (value * gamma).bfloat16().float()
            lo, hi = value.chunk(2, dim=-1)
            rotated = torch.cat((lo * cos - hi * sin, hi * cos + lo * sin), dim=-1)
            return rotated.bfloat16().flatten(1)

        q = norm_rope(q, p["q_norm_weight"], p["q_epsilon"])
        k = norm_rope(k, p["k_norm_weight"], p["k_epsilon"])
        parts = [t.reshape(x.size(0), c.sp_size, -1) for t in (q, k, v)]
        send = torch.cat(parts, dim=-1).transpose(0, 1).contiguous()
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=self.group)
        output = received.reshape(x.size(0) * c.sp_size, -1)
        return tuple(
            t.contiguous()
            for t in output.split([w // c.sp_size for w in widths], dim=-1)
        )

    def close(self):
        dist.destroy_process_group(self.group)
