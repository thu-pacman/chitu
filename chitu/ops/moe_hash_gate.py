# SPDX-FileCopyrightText: 2025 Qingcheng.AI
# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


@make_op_dispatcher
def moe_hash_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    topk: int,
    *,
    score_func: str,
    impl: str = "auto",
):
    raise NotImplementedError


@moe_hash_gate.register_auto
def _auto_moe_hash_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    topk: int,
    *,
    score_func: str,
):
    if has_chitu_backend and x.is_cuda and score_func == "sqrtsoftplus" and topk == 6:
        return "cuda"
    return "torch"


@moe_hash_gate.register("torch")
def moe_hash_gate_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    topk: int,
    *,
    score_func: str,
):
    scores = F.linear(x.float(), weight.float())
    if score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif score_func == "sigmoid":
        scores = scores.sigmoid()
    elif score_func == "sqrtsoftplus":
        scores = F.softplus(scores).sqrt()
    else:
        raise ValueError(f"Unsupported score function: {score_func}")

    indices = tid2eid[input_ids.to(torch.long)].long()
    weights = scores.gather(1, indices)
    if score_func != "softmax":
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, indices


@moe_hash_gate.register("cuda", available=has_chitu_backend)
def moe_hash_gate_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    topk: int,
    *,
    score_func: str,
):
    if score_func != "sqrtsoftplus":
        raise NotImplementedError("CUDA hash gate only supports sqrtsoftplus now.")
    if topk != 6:
        raise NotImplementedError("CUDA hash gate only supports topk=6 now.")

    bs = x.shape[0]
    indices = torch.empty(bs, topk, dtype=torch.int32, device=x.device)
    weights = torch.empty(bs, topk, dtype=torch.float32, device=x.device)
    chitu_backend.cuda_hash_route_gate(
        x.contiguous(),
        weight.contiguous(),
        input_ids.contiguous(),
        tid2eid.contiguous(),
        indices,
        weights,
        topk,
    )
    return weights, indices.long()
