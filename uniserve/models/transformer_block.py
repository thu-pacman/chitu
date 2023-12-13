import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformer_2d import (
    Transformer2DModel,
    BasicTransformerBlock,
)
from diffusers.models.attention_processor import (
    Attention,
)
from flash_attn import (
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_func,
)
import uniserve
import uniserve.layers as unn
from typing import Optional, Dict, Any


class BatchedLinear_nhwc(nn.Module):
    def __init__(self, batches, in_features, out_features):
        super().__init__()
        self.batches = batches
        self.in_features = in_features
        self.out_feature = out_features
        self.weight = nn.Parameter(torch.zeros(batches, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(batches, out_features))

    def forward(self, input_tensor):
        assert self.batches == input_tensor.shape[0]
        output = torch.bmm(input_tensor, self.weight)
        return output + self.bias.unsqueeze(1)

    # # Example
    # self.to_kv = BatchedLinear_nhwc(
    #     2, shadow.to_k.in_features, shadow.to_k.out_features
    # )
    # data1 = shadow.to_k.weight.data
    # data2 = shadow.to_v.weight.data
    # self.to_kv.weight.data = torch.stack([data1.t(), data2.t()], dim=0)
    # if shadow.to_k.bias is not None:
    #     assert shadow.to_v.bias is not None
    #     bias1 = shadow.to_k.bias.data
    #     bias2 = shadow.to_v.bias.data
    #     self.to_kv.bias.data = torch.stack([bias1, bias2], dim=0)


class RaggedAttentionblock_nhwc(nn.Module):
    def __init__(self, shadow: Attention, enco=False):
        super().__init__()
        self.inner_dim = shadow.inner_dim
        self.cross_attention_dim = shadow.cross_attention_dim
        self.dropout = shadow.dropout
        self.enco = enco
        self.to_out = shadow.to_out[0]
        self.heads = shadow.heads

        has_bias = shadow.to_q.bias is not None
        assert has_bias == (shadow.to_k.bias is not None)
        assert has_bias == (shadow.to_v.bias is not None)
        if enco:  # cross-attention
            self.to_q = shadow.to_q
            self.to_kv = torch.nn.Linear(
                shadow.to_q.in_features, 2 * shadow.to_q.out_features, bias=has_bias
            )
            self.to_kv.weight.data = torch.concat(
                [shadow.to_k.weight.data, shadow.to_v.weight.data], dim=0
            )
            if has_bias:
                self.to_kv.bias.data = torch.concat(
                    [shadow.to_k.bias.data, shadow.to_v.bias.data], dim=0
                )
        else:
            self.to_qkv = torch.nn.Linear(
                shadow.to_q.in_features, 3 * shadow.to_q.out_features, bias=has_bias
            )
            self.to_qkv.weight.data = torch.concat(
                [
                    shadow.to_q.weight.data,
                    shadow.to_k.weight.data,
                    shadow.to_v.weight.data,
                ],
                dim=0,
            )
            if has_bias:
                self.to_qkv.bias.data = torch.concat(
                    [
                        shadow.to_q.bias.data,
                        shadow.to_k.bias.data,
                        shadow.to_v.bias.data,
                    ],
                    dim=0,
                )

    def forward(
        self,
        q,
        k,
        v,
        cu_seqlens,
        max_len,
    ):
        nheads = self.heads
        head_dims = self.inner_dim // nheads  # hidden size per head
        if self.enco:  # cross attention
            assert k is v
            Q = self.to_q(q).reshape(-1, nheads, head_dims)
            # KV = self.to_kv(torch.stack([k, v], dim=0)).reshape(
            #     2, -1, nheads, head_dims
            # )
            KV = self.to_kv(k).reshape(-1, 2, nheads, head_dims)
            batches = cu_seqlens.shape[0]
            out = torch.ops.uniserve.flashattn_varlen_fwd(
                Q,
                KV[:, 0],
                KV[:, 1],
                cu_seqlens,
                torch.tensor(
                    range(0, batches * 77, 77), dtype=torch.int32, device="cuda"
                ),
                max_len,
                77,
            )
        else:  # self attention
            assert q is k and k is v
            QKV = self.to_qkv(q).reshape(-1, 3, nheads, head_dims)
            # QKV: [seq, 3, #heads, head_dim]
            out = torch.ops.uniserve.flashattn_varlen_fwd(
                QKV[:, 0],
                QKV[:, 1],
                QKV[:, 2],
                cu_seqlens,
                cu_seqlens,
                max_len,
                max_len,
            )

        out = out.reshape(-1, self.inner_dim)

        out = self.to_out(out)
        return out


class RaggedTransformerBlock_nhwc(nn.Module):
    def __init__(self, shadow: BasicTransformerBlock):
        assert isinstance(shadow, BasicTransformerBlock)
        super().__init__()

        self.shadow = shadow
        self.norm1 = shadow.norm1
        self.norm2 = shadow.norm2
        self.norm3 = shadow.norm3
        ## No LoRA version
        self.attn1 = RaggedAttentionblock_nhwc(shadow.attn1)
        self.attn2 = RaggedAttentionblock_nhwc(shadow.attn2, True)
        # self.scaled_dpa = unn.RaggedNseqfAttentionForward()
        self.ff = shadow.ff
        assert self.attn1.inner_dim % self.attn1.heads == 0
        assert self.attn2.inner_dim % self.attn2.heads == 0

    def forward(
        self,
        input_tensor: torch.Tensor,
        cum_index_cuda: torch.Tensor,
        idx1d_cpu: torch.Tensor,  # to calculate the max length for FA2
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
        max_length = int(torch.max(idx1d_cpu))
        hidden_states = input_tensor
        ## attention 1
        residual_states = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn1(
            hidden_states, hidden_states, hidden_states, cum_index_cuda, max_length
        )
        hidden_states += residual_states

        ## attention 2
        residual_states = hidden_states
        encoder_hidden_states = encoder_hidden_states.reshape(-1, 2048)
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            cum_index_cuda,
            max_length,
        )
        hidden_states += residual_states

        ## feedforward
        residual_states = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(hidden_states, scale=1.0)
        hidden_states += residual_states

        return hidden_states
