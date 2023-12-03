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
        if shadow.conv_shortcut is not None:
            self.conv_shortcut = unn.RaggedNhwcConv2d(shadow.conv_shortcut)
        else:
            self.conv_shortcut = None
        self.nonlinearity = shadow.nonlinearity
        self.time_emb_proj = shadow.time_emb_proj
        self.output_scale_factor = shadow.output_scale_factor
        self.in_channels = shadow.in_channels
        self.out_channels = shadow.out_channels

    def norm_act_conv_nchw2nhwc(self, x, c, idx_cpu, norm, nonlinearity, conv):
        x = norm(x, c, idx_cpu)
        x = torch.ops.uniserve.ragged_nchw_to_nhwc(x, c, idx_cpu)  # [nhw, c]
        x = nonlinearity(x)
        x = conv(x, c, idx_cpu)
        return x

    def forward(
        self,
        input_tensor: torch.Tensor,
        c: int,
        idx_cuda: torch.Tensor,
        idx_cpu: torch.Tensor,
        temb: torch.Tensor,
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
        c0 = c

        x = self.norm_act_conv_nchw2nhwc(
            x, c, idx_cpu, self.norm1, self.nonlinearity, self.conv1
        )
        c = self.conv1.out_channels

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb, scale)  # [n, 1280]

        # x = x + temb
        x = torch.ops.uniserve.addB_jr_rr(x, idx_cuda, temb)

        x = torch.ops.uniserve.ragged_nhwc_to_nchw(x, c, idx_cpu)  # [nchw]

        x = self.norm_act_conv_nchw2nhwc(
            x, c, idx_cpu, self.norm2, self.nonlinearity, self.conv2
        )

        # TOOD: to be optimized
        if self.conv_shortcut is not None:
            # nchw -> nhwc -> nchw
            input_tensor = torch.ops.uniserve.ragged_nchw_to_nhwc(
                input_tensor, c0, idx_cpu
            )
            input_tensor = self.conv_shortcut(input_tensor, c0, idx_cpu)
            input_tensor = torch.ops.uniserve.ragged_nhwc_to_nchw(
                input_tensor, c, idx_cpu
            )

        x = torch.ops.uniserve.ragged_nhwc_to_nchw(x, c, idx_cpu)  # [nchw]

        x = (input_tensor + x) / self.output_scale_factor

        return x


class RaggedResnetBlock2D_nhwc(RaggedResnetBlock2D_nchw):
    def __init__(self, shadow: ResnetBlock2D):
        super().__init__(shadow)
        self.shadow = shadow

    def forward(
        self,
        input_tensor: torch.Tensor,
        c: int,
        idx_cuda: torch.Tensor,
        idx_cpu: torch.Tensor,
        temb: torch.Tensor,
        scale: float = 1.0,
    ):
        x = input_tensor
        # TOOD: to be optimized
        x = torch.ops.uniserve.ragged_nhwc_to_nchw(x, c, idx_cpu)
        x = super().forward(x, c, idx_cuda, idx_cpu, temb, scale)
        x = torch.ops.uniserve.ragged_nchw_to_nhwc(x, self.shadow.out_channels, idx_cpu)
        return x
