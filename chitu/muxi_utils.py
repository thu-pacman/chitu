from typing import List, Optional, Union
from typing_extensions import override
from dataclasses import dataclass
import functools
import torch

from chitu.utils import try_import_opt_dep
from chitu.quantization import (
    NormalLinear,
    Blockfp8Linear,
    NormalMoeExperts,
    Blockfp8MoeExperts,
)
from chitu.native_layout import (
    enable_native_layout_weight,
    NativeLayoutTensor,
    Vector,
    BatchPaddedActivation,
)

muxi_layout_kernels, has_muxi_layout_kernels = try_import_opt_dep(
    "muxi_layout_kernels", "muxi_layout_kernels"
)
tbsgemm, has_tbsgemm = try_import_opt_dep("tbsgemm", "muxi_w8a8_kernels")


class MuxiNativeLayoutActivation(NativeLayoutTensor):
    @classmethod
    @override
    def convert_from(
        cls, tensor: BatchPaddedActivation
    ) -> "MuxiNativeLayoutActivation":
        # NOTE: @functools.singledispatchmethod has a bug in Python 3.8
        # (https://stackoverflow.com/questions/62696796/singledispatchmethod-and-class-method-decorators-in-python-3-8)
        # Use `if` for now

        if isinstance(tensor, BatchPaddedActivation):
            assert tensor.multiple_of == 16
            return cls(
                tensor.plain_shape, muxi_layout_kernels.layoutB(tensor.layout_tensor)
            )
        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to MuxiNativeLayoutActivation"
            )


class MuxiNativeLayoutWeight(NativeLayoutTensor):
    @classmethod
    @override
    def convert_from(cls, tensor: torch.Tensor) -> "MuxiNativeLayoutWeight":
        # NOTE: @functools.singledispatchmethod has a bug in Python 3.8
        # (https://stackoverflow.com/questions/62696796/singledispatchmethod-and-class-method-decorators-in-python-3-8)
        # Use `if` for now

        if isinstance(tensor, torch.Tensor):
            m, k = tensor.shape
            assert m % 128 == 0
            assert k % 128 == 0
            return cls(
                tensor.shape,
                tensor.reshape(m // 16, 16, k // 8, 8).permute(0, 2, 1, 3).contiguous(),
            )
        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to MuxiNativeLayoutWeight"
            )


class MuxiNativeLayoutGroupWeight(NativeLayoutTensor):
    @classmethod
    @override
    def convert_from(cls, tensor: torch.Tensor) -> "MuxiNativeLayoutGroupWeight":
        # NOTE: @functools.singledispatchmethod has a bug in Python 3.8
        # (https://stackoverflow.com/questions/62696796/singledispatchmethod-and-class-method-decorators-in-python-3-8)
        # Use `if` for now

        if isinstance(tensor, torch.Tensor):
            e, m, k = tensor.shape
            assert m % 128 == 0
            assert k % 128 == 0
            return cls(
                tensor.shape,
                tensor.reshape(e, m // 16, 16, k // 8, 8)
                .permute(0, 1, 3, 2, 4)
                .contiguous(),
            )
        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to MuxiNativeLayoutGroupWeight"
            )


@functools.singledispatch
def linear_muxi_layout_native_y(
    x: Union[torch.Tensor, Vector, BatchPaddedActivation, MuxiNativeLayoutActivation],
    w: MuxiNativeLayoutWeight,
    b=None,
) -> Union[Vector, MuxiNativeLayoutActivation]:
    raise ValueError(f"Unsupported input type: {type(x)}")


@linear_muxi_layout_native_y.register
def _(
    x: torch.Tensor, w: MuxiNativeLayoutWeight, b=None
) -> Union[Vector, MuxiNativeLayoutActivation]:
    if x.numel() == x.shape[-1]:
        x = Vector.convert_from(x)
    else:
        x = BatchPaddedActivation.convert_from(x, multiple_of=16)
    return linear_muxi_layout_native_y(x, w, b)


@linear_muxi_layout_native_y.register
def _(
    x: BatchPaddedActivation, w: MuxiNativeLayoutWeight, b=None
) -> MuxiNativeLayoutActivation:
    x_transposed = MuxiNativeLayoutActivation.convert_from(x)
    return linear_muxi_layout_native_y(x_transposed, w, b)


@linear_muxi_layout_native_y.register
def _(x: Vector, w: MuxiNativeLayoutWeight, b=None) -> Vector:
    out_features, in_features = w.plain_shape
    out_shape = list(x.plain_shape[:-1]) + [out_features]

    return Vector.convert_from(
        muxi_layout_kernels.gemv_layoutA(
            w.layout_tensor, x.layout_tensor.view(1, -1), bias=b
        ).view(out_shape)
    )


@linear_muxi_layout_native_y.register
def _(
    x_transposed: MuxiNativeLayoutActivation, w: MuxiNativeLayoutWeight, b=None
) -> MuxiNativeLayoutActivation:
    out_features, in_features = w.plain_shape
    out_shape = list(x_transposed.plain_shape[:-1]) + [out_features]

    if x_transposed.layout_tensor.shape[1] * 16 > 256:
        y = BatchPaddedActivation(
            out_shape,
            muxi_layout_kernels.muxi_hgemm_layout(
                w.layout_tensor, x_transposed.layout_tensor, bias=b
            ),
            multiple_of=x.multiple_of,
        )
        y = MuxiNativeLayoutActivation.convert_from(y)
    else:
        y = MuxiNativeLayoutActivation(
            out_shape,
            muxi_layout_kernels.gemm_layoutABC(
                w.layout_tensor, x_transposed.layout_tensor, bias=b
            ),
        )
    return y


@functools.singledispatch
def linear_muxi_layout_contig_y(
    x_transposed: Union[
        torch.Tensor, Vector, BatchPaddedActivation, MuxiNativeLayoutActivation
    ],
    w: MuxiNativeLayoutWeight,
    b=None,
) -> Union[Vector, BatchPaddedActivation]:
    raise ValueError(f"Unsupported input type: {type(x_transposed)}")


@linear_muxi_layout_contig_y.register
def _(
    x: torch.Tensor, w: MuxiNativeLayoutWeight, b=None
) -> Union[Vector, BatchPaddedActivation]:
    if x.numel() == x.shape[-1]:
        x = Vector.convert_from(x)
    else:
        x = BatchPaddedActivation.convert_from(x, multiple_of=16)
    return linear_muxi_layout_contig_y(x, w, b)


@linear_muxi_layout_contig_y.register
def _(x_transposed: Vector, w: MuxiNativeLayoutWeight, b=None) -> Vector:
    out_features, in_features = w.plain_shape
    out_shape = list(x_transposed.plain_shape[:-1]) + [out_features]

    return Vector.convert_from(
        muxi_layout_kernels.gemv_layoutA(
            w.layout_tensor, x_transposed.layout_tensor.view(1, -1), bias=b
        ).view(out_shape)
    )


@linear_muxi_layout_contig_y.register
def _(
    x_transposed: MuxiNativeLayoutActivation, w: MuxiNativeLayoutWeight, b=None
) -> BatchPaddedActivation:
    out_features, in_features = w.plain_shape
    out_shape = list(x_transposed.plain_shape[:-1]) + [out_features]

    if x_transposed.layout_tensor.shape[1] * 16 > 256:
        y = BatchPaddedActivation(
            out_shape,
            muxi_layout_kernels.muxi_hgemm_layout(
                w.layout_tensor, x_transposed.layout_tensor, bias=b
            ),
            multiple_of=16,
        )
    else:
        y = BatchPaddedActivation(
            out_shape,
            muxi_layout_kernels.gemm_layoutAB_ContinuousC(
                w.layout_tensor, x_transposed.layout_tensor, bias=b
            ),
            multiple_of=16,
        )
    return y


@linear_muxi_layout_contig_y.register
def _(
    x: BatchPaddedActivation, w: MuxiNativeLayoutWeight, b=None
) -> BatchPaddedActivation:
    out_features, in_features = w.plain_shape
    out_shape = list(x.plain_shape[:-1]) + [out_features]

    bs = x.layout_tensor.shape[0]
    assert bs % 16 == 0
    if bs < 128:
        y = BatchPaddedActivation(
            out_shape,
            muxi_layout_kernels.gemm_layoutA_linear(
                w.layout_tensor, x.layout_tensor, bias=b
            ),
            multiple_of=x.multiple_of,
        )
    else:
        m, k = w.plain_shape
        n, k = x.layout_tensor.shape
        y = BatchPaddedActivation(
            out_shape,
            muxi_layout_kernels.gemm_layoutA_wapper(
                w.layout_tensor,
                x.layout_tensor,
                m,
                n,
                k,
                alpha=1,
                beta=0,
                kernelParam1=128,
                kernelParam2=128,
                kernelParam3=128,
                kernelId=2,
                bias=b,
            ),
            multiple_of=x.multiple_of,
        )
    return y


@functools.singledispatch
def blockfp8_linear_muxi_layout_contig_y(
    x: Union[torch.Tensor, Vector, BatchPaddedActivation],
    w: MuxiNativeLayoutGroupWeight,
    b=None,
    weight_scale=None,
) -> Union[Vector, BatchPaddedActivation]:
    raise ValueError(f"Unsupported input type: {type(x)}")


@blockfp8_linear_muxi_layout_contig_y.register
def _(
    x: torch.Tensor,
    w: MuxiNativeLayoutGroupWeight,
    b=None,
    weight_scale=None,
) -> Union[Vector, BatchPaddedActivation]:
    if x.numel() == x.shape[-1]:
        x = Vector.convert_from(x)
    else:
        x = BatchPaddedActivation.convert_from(x, multiple_of=16)
    return blockfp8_linear_muxi_layout_contig_y(x, w, b, weight_scale)


@blockfp8_linear_muxi_layout_contig_y.register
def _(
    x: Vector,
    w: MuxiNativeLayoutGroupWeight,
    b=None,
    weight_scale=None,
) -> Vector:
    out_features, in_features = w.plain_shape
    out_shape = list(x.plain_shape[:-1]) + [out_features]

    assert weight_scale is not None

    return Vector.convert_from(
        muxi_layout_kernels.gemv_layoutA(
            w.layout_tensor,
            x.layout_tensor.view(1, -1),
            scale_matrix=weight_scale,
            bias=b,
        ).view(out_shape)
    )


@blockfp8_linear_muxi_layout_contig_y.register
def _(
    x: BatchPaddedActivation,
    w: MuxiNativeLayoutGroupWeight,
    b=None,
    weight_scale=None,
) -> BatchPaddedActivation:
    out_features, in_features = w.plain_shape
    out_shape = list(x.plain_shape[:-1]) + [out_features]

    assert weight_scale is not None

    bs = x.layout_tensor.shape[0]
    assert bs % 16 == 0
    if bs < 128:
        y = BatchPaddedActivation(
            out_shape,
            muxi_layout_kernels.gemm_layoutA_linear(
                w.layout_tensor, x.layout_tensor, scale_matrix=weight_scale, bias=b
            ),
            multiple_of=x.multiple_of,
        )
    else:
        m, k = w.plain_shape
        n, k = x.layout_tensor.shape
        y = BatchPaddedActivation(
            out_shape,
            muxi_layout_kernels.gemm_layoutA_soft_fp8_wapper(
                w.layout_tensor,
                weight_scale,
                x.layout_tensor,
                m,
                n,
                k,
                alpha=1,
                beta=0,
                kernelParam1=128,
                kernelParam2=128,
                kernelParam3=128,
                kernelId=2,
                bias=b,
            ),
            multiple_of=x.multiple_of,
        )
    return y


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
):

    assert (
        scoring_func == "softmax" or scoring_func == "sigmoid"
    ), "Only softmax and sigmoid are supported now"
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if num_expert_group is None:
        num_expert_group = 1
    if topk_group is None:
        topk_group = 1

    B, H = hidden_states.shape

    expertsIds = torch.empty(B, topk, dtype=torch.int32, device=hidden_states.device)
    selected_experts_weights = torch.empty(
        B, topk, dtype=hidden_states.dtype, device=hidden_states.device
    )

    score_fun = 0
    if scoring_func == "softmax":
        score_fun = 0
    elif scoring_func == "sigmoid":
        score_fun = 1
    else:
        raise ValueError("Unsupported scoring function")

    muxi_layout_kernels.fused_routing_gate(
        gating_output,
        score_fun,
        B,
        H,
        num_expert_group,
        topk_group,
        expertsIds,
        selected_experts_weights,
        topk,
        e_score_correction_bias,
    )

    if renormalize:
        selected_experts_weights = (
            selected_experts_weights
            / selected_experts_weights.sum(dim=-1, keepdim=True)
        )

    return selected_experts_weights, expertsIds


def muxi_fused_experts(
    hidden_states: torch.Tensor,
    w1: MuxiNativeLayoutGroupWeight,
    w2: MuxiNativeLayoutGroupWeight,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    soft_fp8: bool = False,
):
    """
    hidden_states: [_, H], needn't be padded to 16.
    w1: [E, 2 * M, K]
    w2: [E, K, M]
    topk_weights: [B, topk]
    topk_ids: [B, topk]
    """

    assert isinstance(w1, MuxiNativeLayoutGroupWeight)
    assert isinstance(w2, MuxiNativeLayoutGroupWeight)

    micro_batchsize = 16

    assert inplace == True, "Only inplace is supported for now"
    assert topk_weights.shape == topk_ids.shape
    # assert topk_weights.shape[0] % 16 == 0

    shape = hidden_states.size()
    B = hidden_states.size(0)

    # 2. Compute the experts output
    e1, m1, k1 = w1.plain_shape
    e2, m2, k2 = w2.plain_shape
    assert e1 == e2
    assert k1 == m2

    topK = topk_weights.size(1)
    max_num_tokens_padded = (topK * B) + e1 * (micro_batchsize - 1)
    sorted_token_ids = torch.empty(
        max_num_tokens_padded, dtype=torch.int32, device="cuda"
    )
    cumsum_buffer = torch.empty(e1 + 1, dtype=torch.int32, device="cuda")
    padded_num_experts = torch.empty(1, dtype=torch.int32, device="cuda")
    experts_ids = torch.empty(
        (max_num_tokens_padded + micro_batchsize - 1) // micro_batchsize,
        dtype=torch.int32,
        device="cuda",
    )
    C = torch.zeros(topK * m1 * B, dtype=hidden_states.dtype, device="cuda")
    y = torch.zeros_like(hidden_states)

    if soft_fp8:
        assert w1_scale is not None and w2_scale is not None
        muxi_layout_kernels.fused_experts_compute(
            w1.layout_tensor,
            w2.layout_tensor,
            hidden_states,
            B,
            e1,
            topk_ids.shape[-1],
            topk_ids,
            topk_weights,
            sorted_token_ids,
            cumsum_buffer,
            padded_num_experts,
            experts_ids,
            C,
            y,
            w1_scale,
            w2_scale,
            block_shape,
            soft_fp8,
        )
    else:
        muxi_layout_kernels.fused_experts_compute(
            w1.layout_tensor,
            w2.layout_tensor,
            hidden_states,
            B,
            e1,
            topk_ids.shape[-1],
            topk_ids,
            topk_weights,
            sorted_token_ids,
            cumsum_buffer,
            padded_num_experts,
            experts_ids,
            C,
            y,
        )
    return y.view(shape)


class LinearMuxiLayoutNativeY(
    enable_native_layout_weight("weight", MuxiNativeLayoutWeight), NormalLinear
):
    def forward(
        self,
        x: Union[
            torch.Tensor, Vector, BatchPaddedActivation, MuxiNativeLayoutActivation
        ],
    ) -> MuxiNativeLayoutActivation:
        return linear_muxi_layout_native_y(
            x, self.get_native_layout_weight(), self.bias
        )


class LinearMuxiLayoutContigY(
    enable_native_layout_weight("weight", MuxiNativeLayoutWeight), NormalLinear
):
    def forward(
        self,
        x: Union[
            torch.Tensor, Vector, BatchPaddedActivation, MuxiNativeLayoutActivation
        ],
    ) -> torch.Tensor:
        return linear_muxi_layout_contig_y(
            x, self.get_native_layout_weight(), self.bias
        ).convert_to_plain()


class Blockfp8LinearMuxiLayoutContigY(
    enable_native_layout_weight("weight", MuxiNativeLayoutWeight), Blockfp8Linear
):
    def forward(
        self, x: Union[torch.Tensor, Vector, BatchPaddedActivation]
    ) -> torch.Tensor:
        return blockfp8_linear_muxi_layout_contig_y(
            x, self.get_native_layout_weight(), self.bias, self.scale
        ).convert_to_plain()


class NormalMoeExpertsMuxiLayout(
    enable_native_layout_weight("gate_up_proj_weight", MuxiNativeLayoutGroupWeight),
    enable_native_layout_weight("down_proj_weight", MuxiNativeLayoutGroupWeight),
    NormalMoeExperts,
):
    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        moe_world_size: int,
        moe_rank: int,
        op_impl: str,
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
        *,
        ############################################
        # Parameters specific to this quantization
        dtype: Optional[torch.dtype] = None,
    ):
        if fuse_shared_experts:
            raise NotImplementedError(
                "Fused shared experts is not supported for muxi_layout_kernels"
            )
        if not merge_gate_up:
            raise NotImplementedError(
                "muxi_layout_kernels for fused MoE requires merge_gate_up=True"
            )
        super().__init__(
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            n_activated_experts=n_activated_experts,
            moe_world_size=moe_world_size,
            moe_rank=moe_rank,
            op_impl=op_impl,
            fuse_shared_experts=fuse_shared_experts,
            checkpoint_prefix=checkpoint_prefix,
            merge_gate_up=merge_gate_up,
        )

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor):
        shape = x.size()
        x = x.view(-1, self.dim)
        y = muxi_fused_experts(
            hidden_states=x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w2=self.get_native_layout_down_proj_weight(),
            topk_weights=weights,
            topk_ids=indices,
            inplace=True,
            block_shape=[128, 128],
            soft_fp8=False,
        )
        return y.view(shape)


class Blockfp8MoeExpertsMuxiLayout(
    enable_native_layout_weight("gate_up_proj_weight", MuxiNativeLayoutGroupWeight),
    enable_native_layout_weight("down_proj_weight", MuxiNativeLayoutGroupWeight),
    Blockfp8MoeExperts,
):
    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        moe_world_size: int,
        moe_rank: int,
        op_impl: str,
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
        ############################################
        # No parameters specific to this quantization
    ):
        if fuse_shared_experts:
            raise NotImplementedError(
                "Fused shared experts is not supported for muxi_layout_kernels"
            )
        if not merge_gate_up:
            raise NotImplementedError(
                "muxi_layout_kernels for fused MoE requires merge_gate_up=True"
            )
        super().__init__(
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            n_activated_experts=n_activated_experts,
            moe_world_size=moe_world_size,
            moe_rank=moe_rank,
            op_impl=op_impl,
            fuse_shared_experts=fuse_shared_experts,
            checkpoint_prefix=checkpoint_prefix,
            merge_gate_up=merge_gate_up,
        )

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        y = muxi_fused_experts(
            hidden_states=x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w2=self.get_native_layout_down_proj_weight(),
            topk_weights=weights,
            topk_ids=indices,
            inplace=True,
            w1_scale=self.gate_up_proj_scale,
            w2_scale=self.down_proj_scale,
            block_shape=[128, 128],
            soft_fp8=True,
        )
        return y.view(shape)
