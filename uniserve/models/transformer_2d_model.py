import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformer_2d import (
    Transformer2DModel,
    BasicTransformerBlock,
)
from .transformer_block import RaggedTransformerBlock_nhwc
import uniserve
import uniserve.layers as unn


class RaggedTransformer2DModel_nchw(nn.Module):
    def __init__(self, shadow: Transformer2DModel):
        assert isinstance(shadow, Transformer2DModel)
        super().__init__()

        self.shadow = shadow
        self.transformer_blocks = []
        for block in shadow.transformer_blocks:
            self.transformer_blocks.append(RaggedTransformerBlock_nhwc(block))

        self.norm = unn.RaggedNchwGroupNorm(shadow.norm)
        self.use_linear_projection = shadow.use_linear_projection
        self.num_attention_heads = shadow.num_attention_heads
        self.attention_head_dim = shadow.attention_head_dim
        self.is_input_continuous = shadow.is_input_continuous
        self.is_input_vectorized = shadow.is_input_vectorized
        self.is_input_patches = shadow.is_input_patches
        if not self.use_linear_projection:
            self.proj_in = unn.RaggedNhwcConv2d(shadow.proj_in)
            self.proj_out = unn.RaggedNhwcConv2d(shadow.proj_out)
        else:
            self.proj_in = shadow.proj_in
            self.proj_out = shadow.proj_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        heads_num: int,
        heads_dim: int,
        c: int,
        idx_cuda: torch.Tensor,
        idx2d_cpu: torch.Tensor,
        cum_idx1d_cuda,
        encoder_hidden_states: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.norm(hidden_states, c, idx2d_cpu)

        hidden_states = torch.ops.uniserve.ragged_nchw_to_nhwc(
            hidden_states, c, idx2d_cpu
        )
        if self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
        else:
            hidden_states = self.proj_in(hidden_states, c, idx2d_cpu)

        hidden_states = hidden_states.reshape(-1, c)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                cum_idx1d_cuda,
                idx2d_cpu[2].reshape(-1, idx2d_cpu[2].shape[0]),
                encoder_hidden_states,
            )

        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states, c, idx2d_cpu)

        hidden_states = torch.ops.uniserve.ragged_nhwc_to_nchw(
            hidden_states, c, idx2d_cpu
        )

        output = hidden_states + residual

        return output
