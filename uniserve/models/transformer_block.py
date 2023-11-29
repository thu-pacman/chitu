import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformer_2d import (
    Transformer2DModel,
    BasicTransformerBlock,
)

import uniserve
import uniserve.layers as unn
from typing import Optional, Dict, Any


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
        assert self.attn1.inner_dim % self.attn1.heads == 0
        assert self.attn2.inner_dim % self.attn2.heads == 0

    def norm_attn_output_nseqf(self, q, k, v, LSeq, attn, enco=False):
        Q = attn.to_q(q)
        K = attn.to_k(k)
        V = attn.to_v(v)
        heads = attn.heads
        features = attn.inner_dim // attn.heads  # TODO -> hidden_per_dead
        out = self.scaled_dpa(
            Q.flatten(),
            K.flatten(),
            V.flatten(),
            heads,
            features,
            LSeq,
            enco,
        )
        out = out.reshape(-1, attn.inner_dim)
        out = attn.to_out[0](out)
        return out

    def forward(
        self,
        input_tensor: torch.Tensor,
        # n: int,  # TODO remove this parameter
        LSeq: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        """_summary_

        Args:
            hidden_states (_type_): 2D input tensor
            n (_type_): batch size
            seqs (list[int]): seq lenth per batch
            encoder_hidden_states (_type_): sdxl hidden states
        """
        assert attention_mask is None
        assert encoder_attention_mask is None
        assert timestep is None
        assert cross_attention_kwargs is None
        assert class_labels is None

        x = input_tensor
        x_res = x
        x = self.norm1(x)
        x = self.norm_attn_output_nseqf(x, x, x, LSeq, self.attn1)
        x += x_res
        y = encoder_hidden_states.reshape(-1, 2048)
        x_res = x
        x = self.norm2(x)
        x = self.norm_attn_output_nseqf(x, y, y, LSeq, self.attn2, True)
        x += x_res
        x_res = x
        x = self.norm3(x)
        x = self.ff(x, scale=1.0)
        x += x_res
        return x
