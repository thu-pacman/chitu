import torch
import os
from perf_layerwise import shapes_to_tensors
from torchperf import cuda_timeit
from typing import Iterable, Optional
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers.models.activations import get_activation
from diffusers.models.attention import AdaGroupNorm
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.resnet import upsample_2d, downsample_2d, Upsample2D, Downsample2D

# from diffusers.models.resnet import ResnetBlock2D

torch.manual_seed(0)
torch.set_default_device("cuda")  # This results in onnx export error
torch.set_default_dtype(torch.float16)
torch._dynamo.config.cache_size_limit = 102400


class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" or
            "ada_group" for a stronger conditioning with scale and shift.
        kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        skip_time_act=False,
        time_embedding_norm="default",  # default, scale_shift, ada_group, spatial
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = torch.nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )

        self.conv1 = LoRACompatibleConv(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = LoRACompatibleLinear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = LoRACompatibleLinear(
                    temb_channels, 2 * out_channels
                )
            elif (
                self.time_embedding_norm == "ada_group"
                or self.time_embedding_norm == "spatial"
            ):
                self.time_emb_proj = None
            else:
                raise ValueError(
                    f"unknown time_embedding_norm : {self.time_embedding_norm} "
                )
        else:
            self.time_emb_proj = None

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = torch.nn.GroupNorm(
                num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
            )

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = LoRACompatibleConv(
            out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1
        )

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(
                    in_channels, use_conv=False, padding=1, name="op"
                )

        self.use_in_shortcut = (
            self.in_channels != conv_2d_out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )

        self.conv_shortcut = None
        assert self.use_in_shortcut == False

    def forward_uniform(self, input_tensor, temb, scale: float = 1.0):
        hidden_states = input_tensor

        assert self.time_embedding_norm == "default"
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        assert self.upsample is None and self.downsample is None
        # checked
        hidden_states = self.conv1(hidden_states, scale)

        assert self.time_emb_proj is not None
        assert self.time_embedding_norm == "default"
        assert not self.skip_time_act

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb, scale)[:, :, None, None]

        # if temb is not None and self.time_embedding_norm == "default":
        hidden_states = hidden_states + temb[0,]

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states, scale)

        hidden_states = (input_tensor + hidden_states) / self.output_scale_factor

        return hidden_states

    def conv_by_gemm(self, x, conv: nn.Conv2d):
        assert x.dim() == 2  # [nhw, crs]
        x = torch.matmul(x, conv.weight.reshape(conv.weight.shape[0], -1).T)  # [nhw,c]
        x = x + conv.bias.flatten()
        return x

    def ragged_nhwc_to_nchw(self, x, n, c, hs, ws, pos):
        x = x.flatten()
        return torch.concat(
            [x[pos[i] : pos[i + 1]].reshape(-1, c).T.flatten() for i in range(n)]
        )

    def ragged_nchw_norm_silu_unfold(self, norm, x, n, c, hs, ws, pos):
        cache = []
        for i in range(n):
            ex = x[pos[i] : pos[i + 1]].reshape(1, c, hs[i], ws[i])
            ex: torch.Tensor = norm(ex)
            # ex.flatten(start_dim=2).flatten(0, 1)  # [c, hw]
            ex = self.nonlinearity(ex)  # [1, c, h, w]
            ex = torch.nn.functional.unfold(ex, 3, 1, 1)  # [1, crs, hw]
            ex = ex.transpose(1, 2).flatten(0, 1)  # [hw, crs]
            cache.append(ex)
        x = torch.concat(cache, dim=0)  # [nhw, crs]
        return x

    def forward(
        self, n, c, hs: list[int], ws: list[int], input_tensor, temb, scale: float = 1.0
    ):
        """
        Return a 1D nchw tensor
        """
        assert self.time_embedding_norm == "default"
        assert self.upsample is None and self.downsample is None
        assert self.time_emb_proj is not None
        assert self.time_embedding_norm == "default"
        assert not self.skip_time_act

        # n, c, h, w = input_tensor.shape
        length = []
        pos = [0]
        for h, w in zip(hs, ws):
            cnt = c * h * w
            length.append(cnt)
            pos.append(pos[-1] + cnt)
        # pos.pop()

        x = input_tensor
        x = self.ragged_nchw_norm_silu_unfold(self.norm1, x, n, c, hs, ws, pos)

        # x = self.conv1(x, scale)
        x = self.conv_by_gemm(x, self.conv1)  # [nhw, c]

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb, scale)
        # temb = temb[:, :, None, None]
        temb = temb[0,].flatten()
        x = x + temb

        x = self.ragged_nhwc_to_nchw(x, n, c, hs, ws, pos)  # [nchw]
        # checked

        x = self.ragged_nchw_norm_silu_unfold(self.norm2, x, n, c, hs, ws, pos)
        x = self.conv_by_gemm(x, self.conv2)  # [nhw, c]

        x = self.ragged_nhwc_to_nchw(x, n, c, hs, ws, pos)  # [nchw]
        x = (input_tensor + x) / self.output_scale_factor

        return x


config = {
    "in_channels": 320,
    "out_channels": 320,
    "conv_shortcut": False,
    "dropout": 0,
    "temb_channels": 1280,
    "groups": 32,
    "groups_out": None,
    "pre_norm": True,
    "eps": 1.00e-05,
    "non_linearity": "silu",
    "skip_time_act": False,
    "time_embedding_norm": "default",
    "kernel": None,
    "output_scale_factor": 1,
    "use_in_shortcut": None,
    "up": False,
    "down": False,
    "conv_shortcut_bias": True,
    "conv_2d_out_channels": None,
}

model = ResnetBlock2D(**config).eval().cuda().half()
print(model)


# Transformer layer
# sym_args = (torch.Size([2, 64, 1280]),)
# sym_kwargs = {
#     "attention_mask": None,
#     "encoder_hidden_states": torch.Size([2, 77, 2048]),
#     "encoder_attention_mask": None,
#     "timestep": None,
#     "cross_attention_kwargs": None,
#     "class_labels": None,
# }


def generate_ragged_data(n, c, hs, ws):
    seperate = []
    for i in range(n):
        seperate.append(torch.randn(1, c, hs[i], ws[i]))
    ragged = torch.concat([t.flatten() for t in seperate])
    return seperate, ragged


def test_uniform_input(model):
    n, c = 4, 320
    hs, ws = [32] * n, [32] * n
    sym_args = (torch.Size([n, c, 32, 32]), torch.Size([2, 1280]))
    sym_kwargs = {"scale": 1.0}

    args = shapes_to_tensors(sym_args)
    kwargs = shapes_to_tensors(sym_kwargs)

    if False:
        model = torch.compile(model)
    x = model.forward(n, c, hs, ws, args[0].flatten(), *args[1:], **kwargs)
    y = model.forward_uniform(*args, **kwargs)
    print("x=\n", x)
    print("y (truth)=\n", y)
    print(torch.allclose(x.flatten(), y.flatten(), 1e-3, 1e-3))


def test_ragged_input(model):
    configs = [
        (4, 320, [32, 32, 64, 64], [32, 32, 64, 64]),
        (6, 320, [32, 32, 64, 64, 65, 65], [32, 32, 64, 64, 65, 65]),
    ]

    def run_uniform(seperate_input, temb):
        ans_seperate = []
        for i in seperate_input:
            ans_seperate.append(model.forward_uniform(i, temb))
        return torch.concat([t.flatten() for t in ans_seperate])

    def run_ragged(n, c, hs, ws, unified_input, temb):
        return model.forward(n, c, hs, ws, unified_input, temb)

    # compile the model
    if True:
        # run_uniform = torch.compile(run_uniform)
        run_ragged = torch.compile(run_ragged, dynamic=True)
        model.forward_uniform = torch.compile(model.forward_uniform, dynamic=True)

    for n, c, hs, ws in configs:
        assert n == len(hs) == len(ws)
        # n, c = 4, 320
        # hs, ws = [32, 32, 64, 64], [32, 32, 64, 64]
        # sym_args = (torch.Size([n, c, 32, 32]), torch.Size([2, 1280]))
        # sym_kwargs = {"scale": 1.0}

        seperate_input, unified_input = generate_ragged_data(n, c, hs, ws)
        temb = torch.randn(2, 1280)

        x = run_ragged(n, c, hs, ws, unified_input, temb)
        y = run_uniform(seperate_input, temb)
        print("x=\n", x)
        print("y (truth)=\n", y)
        compiled = False
        print(torch.allclose(x.flatten(), y.flatten(), 1e-3, 1e-3))
        n_close = torch.isclose(x.flatten(), y.flatten(), 1e-3, 1e-3).sum()
        n_el = x.numel()
        print(f"{n_close}/{n_el} {float(n_close)/n_el*100:.1f}%")

        continue
        torch.cuda.profiler.start()
        print(
            f"{cuda_timeit(partial(run_ragged, n, c, hs, ws, unified_input, temb), compile=compiled)}"
        )
        print(
            f"{cuda_timeit(partial(run_uniform, seperate_input, temb), compile=compiled)}"
        )


if __name__ == "__main__":
    test_ragged_input(model)
    # test_uniform_input(model)


# fn = f"b.onnx"
# torch.onnx.export(model, (*args, kwargs), fn, verbose=False, do_constant_folding=False)
# export_output = torch.onnx.dynamo_export(model, *args, *kwargs).save(fn)

# for batch in [1]:
#     print(sym_args, sym_kwargs)
#     args = shapes_to_tensors(sym_args, 2, 2 * batch)
#     kwargs = shapes_to_tensors(sym_kwargs, 2, 2 * batch)
#     model(*args, **kwargs)
#     # fn = f"sdxl_dedup_2.{name}.onnx"
#     # torch default device CUDA makes the export fail
#     torch.onnx.export(model, (*args, kwargs), fn, verbose=True)
#     # infer_onnx(fn)
