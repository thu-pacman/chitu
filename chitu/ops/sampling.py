# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List

import torch

from chitu.utils import try_import_platform_dep
from chitu.device_list import DeviceList

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")

if has_triton:
    from chitu.ops.triton_ops import apply_frequency_penalty_triton


def multinomial(
    probs: torch.Tensor,
    num_samples: int,
    seq_groups: Optional[List] = None,
    impl: str = "torch",
) -> torch.Tensor:
    if impl == "torch":
        return torch.multinomial(probs, num_samples)
    elif impl == "sync-free":
        # Adapted from
        # https://github.com/vllm-project/vllm/blob/4577fc9abb064d74b2082ffc5005cbb82ca91766/vllm/model_executor/layers/sampler.py#L527
        # SPDX-SnippetBegin
        # SPDX-License-Identifier: Apache-2.0
        # SPDX-SnippetCopyrightText: 2025 vLLM Team
        # SDPX—SnippetName: _multinomial from vllm

        if num_samples > 1:
            probs = probs.repeat_interleave(num_samples, dim=0)
        q = torch.empty_like(probs)
        if seq_groups is None:
            q.exponential_()
        else:
            sample_idx = 0
            for seq_group in seq_groups:
                seq_ids = seq_group.seq_ids
                stride = len(seq_ids) * num_samples
                assert seq_group.generator is not None
                q[sample_idx : sample_idx + stride].exponential_(
                    generator=seq_group.generator
                )
                sample_idx += stride
        # SPDX-SnippetEnd
        return probs.div_(q).argmax(dim=1).view(-1, num_samples)
    else:
        raise NotImplementedError(f"unsupport impl: {impl}")


@torch.no_grad()
def apply_frequency_penalty(
    logits: torch.Tensor,
    logits_index: DeviceList,
    response_list: List[DeviceList],
    response_len_list: DeviceList,
    frequency_penalty: torch.Tensor,
    impl="auto",
):
    bs = len(logits_index)
    if bs == 0:
        return
    assert (
        len(response_list) == bs
        and len(response_len_list) == bs
        and frequency_penalty.shape[0] == bs
    )
    assert frequency_penalty.is_contiguous()
    if impl == "auto":
        # NOTE: This is a temporary solution based tests on h20.
        if has_triton and bs > 8 and bs <= 16:
            impl = "triton"
        elif bs < 16 or has_torch_npu:
            impl = "torch"
        else:
            impl = "cuda"
    if impl == "triton":
        apply_frequency_penalty_triton(
            logits,
            logits_index.to_tensor(),
            response_list,
            response_len_list.to_tensor(),
            frequency_penalty,
        )
    elif impl == "torch":
        for i, idx in enumerate(logits_index.to_tensor()):
            logits[idx].index_add_(
                -1,
                response_list[i].to_tensor(),
                -frequency_penalty[idx]
                * torch.ones(
                    (response_len_list[i],),
                    dtype=logits.dtype,
                    device=logits.device,
                ),
            )
    elif impl == "cuda":
        assert logits.dtype == torch.float
        responses = [response.to_tensor().data_ptr() for response in response_list]
        response_ptr_list = torch.tensor(
            responses, dtype=torch.int64, device=logits.device
        )
        chitu_backend.cuda_frequency_penalty(
            logits,
            logits_index.to_tensor(),
            response_ptr_list,
            frequency_penalty,
            response_len_list.to_tensor(),
            bs,
            logits.shape[-1],
            logits.stride(0),
            logits.stride(1),
        )
    else:
        raise NotImplementedError(f"{impl=}")


@torch.no_grad()
def response_append_cuda(
    response_list,
    tokens_list,
    response_len,
    response_capacity,
    task_num,
):
    assert response_list.dtype == torch.long, f"{response_list.dtype=}"
    assert tokens_list.dtype == torch.long, f"{tokens_list.dtype=}"
    assert response_len.dtype == torch.int, f"{response_len.dtype=}"
    assert response_capacity.dtype == torch.int, f"{response_capacity.dtype=}"
    need_expand = response_len == response_capacity
    new_response_list = torch.empty_like(response_list)
    return_response_list = []

    expand_cpu = need_expand.cpu().tolist()

    new_response_index = []
    new_response_ptr = []
    new_response_capacity = []

    for i in range(task_num):
        if expand_cpu[i]:
            new_len = max(2 * response_capacity[i], 32)
            new_response = torch.empty(
                new_len, dtype=torch.long, device=response_list.device
            )
            new_response_ptr.append(new_response.data_ptr())
            new_response_index.append(i)
            new_response_capacity.append(new_len)
            return_response_list.append((i, new_response))
    if len(new_response_ptr) > 0:
        new_response_list[new_response_index] = torch.tensor(
            new_response_ptr,
            device=new_response_list.device,
            dtype=new_response_list.dtype,
        )
        response_capacity[new_response_index] = torch.tensor(
            new_response_capacity,
            device=response_capacity.device,
            dtype=response_capacity.dtype,
        )
    chitu_backend.cuda_response_append(
        response_list, new_response_list, tokens_list, response_len, need_expand
    )
    return return_response_list


def response_append(tasks, tokens, impl="auto"):
    if impl == "auto":
        if tasks.num_tasks > 8 and has_chitu_backend:
            impl = "cuda"
        else:
            impl = "torch"
    if impl == "torch":
        for it, task in enumerate(tasks.tasks):
            task.response.append(tokens[it])
    elif impl == "cuda":
        new_response = response_append_cuda(
            tasks.response_ptr,
            tokens,
            tasks.response_len,
            tasks.response_capacity,
            task_num=tasks.num_tasks,
        )
        for idx, response in new_response:
            tasks.tasks[idx].response._data = response
            tasks.response_ptr[idx] = response.data_ptr()
        for task in tasks.tasks:
            task.response._len += 1
        tasks.response_len += 1
    else:
        raise NotImplementedError(f"{impl=}")
