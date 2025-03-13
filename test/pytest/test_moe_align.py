import torch
import pytest

from chitu.device_type import is_nvidia


@pytest.mark.skipif(not is_nvidia(), reason="Only NVIDIA GPUs are supported")
def test_moe_align_block_size_cuda():
    # Import inside the `skipif` guard
    from chitu_backend import cuda_moe_align_block_size

    topk_ids = torch.randint(0, 256, (1000,), device="cuda:0")
    num_experts = 256
    block_size = 64
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    sorted_ids_triton = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids_triton.fill_(topk_ids.numel())
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    # Expert ids must be zeroed out to prevent index out of bounds error while
    # mapping global expert ids to local expert ids in expert parallelism.
    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.zeros(
        (num_experts + 1,), dtype=torch.int32, device=topk_ids.device
    )
    cuda_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
    )
    print("num_tokens_post_pad:", num_tokens_post_pad)
    print("topk_ids:", topk_ids)
    # Find indices where topk_ids value is 0
    # Randomly select 10 expert IDs to check
    unique_expert_ids = torch.unique(topk_ids)
    selected_experts = unique_expert_ids
    print(f"Randomly selected experts to check: {selected_experts.tolist()}")

    # For each selected expert, find indices where topk_ids equals that expert
    for expert_id in selected_experts:
        expert_indices = torch.nonzero(topk_ids == expert_id).squeeze()
        # Ensure expert_indices is always iterable (handle both empty tensor and scalar tensor cases)
        if expert_indices.dim() == 0 and expert_indices.numel() > 0:
            expert_indices = expert_indices.unsqueeze(0)  # Convert scalar to 1D tensor
        elif expert_indices.numel() == 0:
            expert_indices = torch.tensor([], dtype=torch.long, device=topk_ids.device)
        print(f"Indices where topk_ids value is {expert_id}:", expert_indices)

        # Get cumsum buffer values for this expert
        start_idx_expert = cumsum_buffer[expert_id].item()
        len_expert = cumsum_buffer[expert_id + 1].item() - start_idx_expert

        print(f"Expert {expert_id}: start_idx={start_idx_expert}, length={len_expert}")

        # Check if any of the expert indices in topk_ids appear in the expert section of sorted_ids
        if len_expert > 0:
            if len(expert_indices) > 0:
                expert_section = sorted_ids[
                    start_idx_expert : start_idx_expert + len_expert
                ]
                for idx in expert_indices:
                    assert idx in expert_section


if __name__ == "__main__":
    test_moe_align_block_size_cuda()
