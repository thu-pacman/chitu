import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformer_2d import (
    Transformer2DModel,
    BasicTransformerBlock,
)

import uniserve
import uniserve.layers as unn


class RaggedTransformerBlock_nhwc(nn.Module):
    def __init__(self, shadow: BasicTransformerBlock):
        assert isinstance(shadow, BasicTransformerBlock)
        super().__init__()

        self.shadow = shadow
        self.norm1 = shadow.norm1
        self.norm2 = shadow.norm2
        self.norm3 = shadow.norm3
        self.attn1 = shadow.attn1
        self.attn2 = shadow.attn2
        self.scaled_dpa = unn.RaggedNseqfAttentionForward()
        self.ff = shadow.ff

    def norm_attn_output_nseqf(
        self, q, k, v, n, LSeq, heads_dim, heads_num, attn, enco=False
    ):
        Q = attn.to_q(q)
        K = attn.to_k(k)
        V = attn.to_v(v)
        out = self.scaled_dpa(
            Q.flatten(), K.flatten(), V.flatten(), heads_num, heads_dim, LSeq, enco
        )
        out = out.reshape(-1, heads_num * heads_dim)
        out = attn.to_out[0](out)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        heads_num: int,
        heads_dim: int,
        n: int,
        LSeq: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        """_summary_

        Args:
            hidden_states (_type_): 2D input tensor
            n (_type_): batch size
            seqs (list[int]): seq lenth per batch
            encoder_hidden_states (_type_): sdxl hidden states
        """
        x_res = hidden_states
        x = self.norm1(hidden_states)
        x = self.norm_attn_output_nseqf(
            x, x, x, n, LSeq, heads_dim, heads_num, self.attn1
        )
        x += x_res
        y = encoder_hidden_states.reshape(-1, 2048)
        x_res = x
        x = self.norm2(x)
        x = self.norm_attn_output_nseqf(
            x, y, y, n, LSeq, heads_dim, heads_num, self.attn2, True
        )
        x += x_res
        x_res = x
        x = self.norm3(x)
        x = self.ff(x, scale=1.0)
        x += x_res
        return x
