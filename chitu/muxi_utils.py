from typing import List, Optional
from .utils import try_import_opt_dep
import torch

from chitu.utils import try_import_opt_dep
from chitu.tensor_parallel import LocalLinear
from chitu.quantization import Blockfp8Linear

muxi_layout_kernels, has_muxi_layout_kernels = try_import_opt_dep(
    "muxi_layout_kernels", "muxi_layout_kernels"
)
tbsgemm, has_tbsgemm = try_import_opt_dep("tbsgemm", "muxi_w8a8_kernels")

muxi_moe_fused, has_muxi_moe_fused = try_import_opt_dep(
    "muxi_moe_fused", "muxi_moe_fused"
)


def preprocess_weights_for_native_layout(
    checkpoint, rpl_names: List[str], cpl_names: List[str]
):
    # NOTE: Currently we skip all the weights that is not part of a layer (which means
    # they are either pre-layers or post-layers). This is only for convenience. They
    # can be supported by native layout in the future.

    def is_rpl_weight(key):
        for name in rpl_names:
            if key.endswith(f".{name}.weight"):  # Not including pre-layer or post-layer
                return True
        return False

    def is_cpl_weight(key):
        for name in cpl_names:
            if key.endswith(f".{name}.weight"):  # Not including pre-layer or post-layer
                return True
        return False

    new_checkpoint = {}
    for key in checkpoint.keys():
        if is_rpl_weight(key):  # Row parallel
            if checkpoint[key].ndim == 3:
                e, m, k = checkpoint[key].shape
                assert m % 128 == 0 and k % 128 == 0
                new_checkpoint[key] = (
                    checkpoint[key]
                    .reshape(e, m // 16, 16, k // 8, 8)
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                    .reshape(e, m, k)
                )
            elif checkpoint[key].ndim == 2:
                m, k = checkpoint[key].shape
                assert m % 128 == 0
                assert k % 128 == 0
                new_checkpoint[key] = (
                    checkpoint[key]
                    .reshape(m // 16, 16, k // 8, 8)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape(m, k)  # Reshape back to for compatibility
                )
            else:
                new_checkpoint[key] = checkpoint[key]
        elif is_cpl_weight(key):  # Column parallel
            if checkpoint[key].ndim == 3:
                e, m, k = checkpoint[key].shape
                assert m % 128 == 0 and k % 128 == 0
                new_checkpoint[key] = (
                    checkpoint[key]
                    .reshape(e, m // 16, 16, k // 8, 8)
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                    .reshape(e, m, k)
                )
            elif checkpoint[key].ndim == 2:
                m, k = checkpoint[key].shape
                assert m % 128 == 0
                assert k % 128 == 0
                new_checkpoint[key] = (
                    checkpoint[key]
                    .reshape(m // 16, 16, k // 8, 8)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape(m, k)  # Reshape back to for compatibility
                )
            else:
                new_checkpoint[key] = checkpoint[key]
        else:
            new_checkpoint[key] = checkpoint[key]
    return new_checkpoint


def linear_layout_contig_x_native_y(x, w, b=None):
    assert x.ndim == 2
    x_is_vector = x.shape[0] == 1
    if not x_is_vector:
        x_transposed = muxi_layout_kernels.layoutB(x)
    # w has already been transposed, but reshaped back for compatibility. We only need to "view" it again.
    w_transposed = w.view(w.shape[0] // 16, w.shape[1] // 8, 16, 8)
    if x_is_vector:
        y = muxi_layout_kernels.gemv_layoutA(w_transposed, x, bias=b)
        # View as 5D to be compatible with "native layout" but make n's tile to be 1.
        y = y.view(y.shape[1] // 32, 1, 4, 1, 8)
    elif x_transposed.shape[1] * 16 > 256:
        y = muxi_layout_kernels.muxi_hgemm_layout(w_transposed, x_transposed, bias=b)
        y = muxi_layout_kernels.layoutB(y)
    else:
        y = muxi_layout_kernels.gemm_layoutABC(w_transposed, x_transposed, bias=b)
    return y


def linear_layout_native_x_contig_y(x_transposed, w, b=None):
    assert x_transposed.ndim == 5
    x_is_vector = x_transposed.shape[1] == 1 and x_transposed.shape[3] == 1
    # w has already been transposed, but reshaped back for compatibility. We only need to "view" it again.
    w_transposed = w.view(w.shape[0] // 16, w.shape[1] // 8, 16, 8)
    if x_is_vector:
        y = muxi_layout_kernels.gemv_layoutA(
            w_transposed, x_transposed.view(1, -1), bias=b
        )
    elif x_transposed.shape[1] * 16 > 256:
        y = muxi_layout_kernels.muxi_hgemm_layout(w_transposed, x_transposed, bias=b)
    else:
        y = muxi_layout_kernels.gemm_layoutAB_ContinuousC(
            w_transposed, x_transposed, bias=b
        )
    return y


def linear_layout_contig_x_contig_y(x, w, b=None):
    assert x.ndim == 2
    bs = x.shape[0]
    x_is_vector = bs == 1

    # w has already been transposed, but reshaped back for compatibility. We only need to "view" it again.
    w_transposed = w.view(w.shape[0] // 16, w.shape[1] // 8, 16, 8)

    if x_is_vector:
        y = muxi_layout_kernels.gemv_layoutA(w_transposed, x, bias=b)
    else:
        assert bs % 16 == 0
        if bs < 128:
            y = muxi_layout_kernels.gemm_layoutA_linear(w_transposed, x, bias=b)
        else:
            m, k = w.shape
            n, k = x.shape
            y = muxi_layout_kernels.gemm_layoutA_wapper(
                w_transposed,
                x,
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
            )
    return y


def blockfp8_linear_layout_contig_x_contig_y(x, w, b=None, weight_scale=None):
    assert x.ndim == 2
    bs = x.shape[0]
    x_is_vector = bs == 1
    w_transposed = w.view(w.shape[0] // 16, w.shape[1] // 8, 16, 8)

    assert weight_scale is not None

    if x_is_vector:
        y = muxi_layout_kernels.gemv_layoutA(
            w_transposed, x, scale_matrix=weight_scale, bias=b
        )
    else:
        assert bs % 16 == 0
        if bs < 128:
            y = muxi_layout_kernels.gemm_layoutA_linear(
                w_transposed, x, scale_matrix=weight_scale, bias=b
            )
        else:
            m, k = w.shape
            n, k = x.shape
            y = muxi_layout_kernels.gemm_layoutA_soft_fp8_wapper(
                w_transposed,
                weight_scale,
                x,
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
            )
    return y


def get_muxi_padded_input(x, dtype=None, bound=16, can_be_single_batch=True):
    assert x.ndim == 2
    bs, seq_len = x.shape
    need_padding = (not can_be_single_batch or bs > 1) and bs % bound != 0
    if need_padding:
        x_padded = torch.zeros(
            (bs + bound - 1) & ~(bound - 1), seq_len, dtype=dtype, device=x.device
        )
        x_padded[:bs, :] = x
        return x_padded
    return x


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

    muxi_moe_fused.fused_routing_gate(
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
    w1: torch.Tensor,
    w2: torch.Tensor,
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
    micro_batchsize = 16

    assert inplace == True, "Only inplace is supported for now"
    assert topk_weights.shape == topk_ids.shape
    # assert topk_weights.shape[0] % 16 == 0

    shape = hidden_states.size()
    B = hidden_states.size(0)

    # 2. Compute the experts output
    e1, m1, k1 = w1.shape
    e2, m2, k2 = w2.shape
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
        muxi_moe_fused.fused_experts_compute(
            w1,
            w2,
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
        muxi_moe_fused.fused_experts_compute(
            w1,
            w2,
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
    del (
        sorted_token_ids,
        cumsum_buffer,
        padded_num_experts,
        experts_ids,
        C,
        hidden_states,
    )

    return y.view(shape)


class LinearLayoutContigXNativeY(LocalLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_layout_contig_x_native_y(x, self.weight, self.bias)


class LinearLayoutNativeXContigY(LocalLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_layout_native_x_contig_y(x, self.weight, self.bias)


class LinearLayoutContigXContigY(LocalLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_layout_contig_x_contig_y(x, self.weight, self.bias)


class Blockfp8LinearLayoutContigXContigY(Blockfp8Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return blockfp8_linear_layout_contig_x_contig_y(
            x, self.weight, self.bias, self.scale
        )
