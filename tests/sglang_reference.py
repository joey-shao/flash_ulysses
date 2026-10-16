"""Real Qwen3Attention + exchange_qkv from the SGLang Ulysses checkout."""

from contextlib import ExitStack

import torch


class SGLangReference:
    def __init__(self, context, seed=1234):
        from sglang.srt.distributed.parallel_state import init_model_parallel_group
        from sglang.srt.distributed.ulysses_parallel import UlyssesRankLayout
        from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
        from sglang.srt.layers.sp.sp_strategy import SPBatchMetadata, exchange_qkv
        from sglang.srt.models.qwen3 import Qwen3Attention
        from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish
        from sglang.srt.server_args import ServerArgs

        self.context = context
        self.metadata_type, self.exchange = SPBatchMetadata, exchange_qkv
        publish(
            ServerArgs(model_path="dummy", tp_size=context.world_size,
                       dtype="bfloat16", device="cuda", bf16_gemm_backend="torch",
                       disable_custom_all_reduce=True),
            role="test", ranks=SpawnRanks(world_rank=context.rank, gpu_id=context.rank),
        )
        initialize_bf16_gemm_config()
        layout = UlyssesRankLayout(context.world_size, context.sp_size)
        group_options = dict(local_rank=context.rank, backend="nccl", use_pynccl=False,
                             use_custom_allreduce=False, use_mscclpp=False,
                             use_torch_symm_mem_allreduce=False)
        self.base_group = init_model_parallel_group(
            layout.base_tp_ranks, group_name="qkv_base_tp", **group_options,
        )
        self.sp_group = init_model_parallel_group(
            layout.sp_ranks, group_name="qkv_sp", **group_options,
        )
        self.scope = ExitStack()
        self.scope.enter_context(get_parallel().override(
            tp_group=self.base_group, tp_rank=self.base_group.rank_in_group,
            tp_size=context.base_tp_size, attn_tp_group=self.base_group,
            attn_tp_rank=self.base_group.rank_in_group, attn_tp_size=context.base_tp_size,
            moe_tp_size=context.base_tp_size, moe_tp_rank=self.base_group.rank_in_group,
        ))
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                self.attention = Qwen3Attention(
                    hidden_size=context.hidden_size, num_heads=context.num_heads,
                    num_kv_heads=context.num_kv_heads, head_dim=128,
                    rms_norm_eps=1e-6, rope_theta=1_000_000,
                    max_position_embeddings=40960,
                    quant_config=None, attention_bias=False,
                ).eval().requires_grad_(False)
        finally:
            torch.set_default_dtype(previous_dtype)
        layer = self.attention
        generator = torch.Generator(device="cuda").manual_seed(
            seed + self.base_group.rank_in_group
        )
        layer.qkv_proj.weight.normal_(std=context.hidden_size ** -0.5, generator=generator)
        layer.q_norm.weight.uniform_(0.5, 1.5, generator=generator)
        layer.k_norm.weight.uniform_(0.5, 1.5, generator=generator)
        # Export exactly the same weights/cache to TK and the PyTorch baseline.
        self.parameters = dict(
            qkv_weight=layer.qkv_proj.weight.t().contiguous(),
            q_norm_weight=layer.q_norm.weight.float().contiguous(),
            k_norm_weight=layer.k_norm.weight.float().contiguous(),
            rope_cache=layer.rotary_emb.cos_sin_cache.float().contiguous(),
            q_epsilon=layer.q_norm.variance_epsilon,
            k_epsilon=layer.k_norm.variance_epsilon,
        )
        layer.qkv_proj.quant_method.process_weights_after_loading(layer.qkv_proj)

    @torch.no_grad()
    def __call__(self, x, positions):
        metadata = self.metadata_type(
            num_tokens=x.size(0) * self.context.sp_size,
            sp_size=self.context.sp_size, sp_rank=self.sp_group.rank_in_group,
        )
        q, k, v = self.attention.forward_prepare_native(positions, x)
        return self.exchange(q, k, v, metadata=metadata, group=self.sp_group)

    def close(self):
        self.scope.close()
        self.sp_group.destroy()
        self.base_group.destroy()
