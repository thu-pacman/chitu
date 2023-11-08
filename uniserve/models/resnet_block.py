import torch
import torch.nn as nn
import torch.nn.functional as F

# from diffusers.models.activations import get_activation
# from diffusers.models.attention import AdaGroupNorm
# from diffusers.models.attention_processor import SpatialNorm
# from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.resnet import (
    upsample_2d,
    downsample_2d,
    Upsample2D,
    Downsample2D,
    ResnetBlock2D,
)

import uniserve
import uniserve.layers as unn

# TODO: why this failed
# import torch.ops.uniserve as uop


class RaggedResnetBlock2D_nchw(nn.Module):
    def __init__(self, shadow: ResnetBlock2D):
        assert isinstance(shadow, ResnetBlock2D)
        super().__init__()
        assert shadow.time_embedding_norm == "default"
        assert shadow.upsample is None and shadow.downsample is None
        assert shadow.time_emb_proj is not None
        assert shadow.time_embedding_norm == "default"
        assert not shadow.skip_time_act

        self.shadow = shadow
        self.norm1 = unn.RaggedNchwGroupNorm(shadow.norm1)
        self.norm2 = unn.RaggedNchwGroupNorm(shadow.norm2)
        self.conv1 = unn.RaggedNhwcConv2d(shadow.conv1)
        self.conv2 = unn.RaggedNhwcConv2d(shadow.conv2)
        self.nonlinearity = shadow.nonlinearity
        self.time_emb_proj = shadow.time_emb_proj
        self.output_scale_factor = shadow.output_scale_factor

    def norm_act_conv_nchw2nhwc(self, x, n, c, hs, ws, HxWs, norm, nonlinearity, conv):
        x = norm(x, n, c, HxWs)
        x = torch.ops.uniserve.ragged_nchw_to_nhwc(x, n, c, HxWs)  # [nhw, c]
        x = nonlinearity(x)
        x = conv(x, n, c, hs, ws, HxWs)
        return x

    def forward(
        self,
        input_tensor,
        n,
        c,
        hs: list[int],
        ws: list[int],
        HxWs: list[int],
        HxWs_tensor: torch.Tensor,
        temb,
        scale: float = 1.0,
    ):
        """
        Return a 1D nchw tensor
        """

        # n, c, h, w = input_tensor.shape
        # length = []
        # pos = [0]
        # for h, w in zip(hs, ws):
        #     cnt = c * h * w
        #     length.append(cnt)
        #     pos.append(pos[-1] + cnt)
        # pos.pop()

        x = input_tensor  # [nchw]
        x = self.norm_act_conv_nchw2nhwc(
            x, n, c, hs, ws, HxWs, self.norm1, self.nonlinearity, self.conv1
        )

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb, scale)  # [n, 1280]

        # x = x + temb
        x = torch.ops.uniserve.addB_jr_rr(x, n, HxWs_tensor, temb)

        x = torch.ops.uniserve.ragged_nhwc_to_nchw(x, n, c, HxWs)  # [nchw]

        x = self.norm_act_conv_nchw2nhwc(
            x, n, c, hs, ws, HxWs, self.norm2, self.nonlinearity, self.conv2
        )
        x = torch.ops.uniserve.ragged_nhwc_to_nchw(x, n, c, HxWs)  # [nchw]

        x = (input_tensor + x) / self.output_scale_factor
        return x
