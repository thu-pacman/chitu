import cinfer_backend, torch
from .triton_kernels import *

"""
def GEMV(x, w):
    # w = w.transpose(1, 0).contiguous()
    output = torch.zeros(x.shape[:-1] + w.shape[-1:], device=x.device, dtype=x.dtype)
    cinfer_backend.gemv(x, w, output)
    return output
"""


@torch.inference_mode()
def torch_attention(query, key, value):
    # print(query.shape, key.shape)
    n_local_kv_heads = key.shape[-2]
    n_local_q_heads = query.shape[-2]
    heads_per_group = n_local_q_heads // n_local_kv_heads
    query = query.view(
        query.shape[0],
        query.shape[1],
        n_local_kv_heads,
        heads_per_group,
        query.shape[-1],
    )
    key = key.view(key.shape[0], key.shape[1], n_local_kv_heads, 1, key.shape[-1])
    value = value.view(
        value.shape[0], value.shape[1], n_local_kv_heads, 1, value.shape[-1]
    )
    if key.shape[-2] < query.shape[-2] and key.shape[-2] == 1:
        new_shape = []
        for i in range(len(key.shape)):
            if i == len(key.shape) - 2:
                new_shape.append(query.shape[-2])
            else:
                new_shape.append(key.shape[i])
        key = key.expand(new_shape)
        value = value.expand(new_shape)
        query = query.view(query.shape[0], query.shape[1], -1, query.shape[-1])
        key = key.reshape(key.shape[0], key.shape[1], -1, key.shape[-1])
        value = value.reshape(value.shape[0], value.shape[1], -1, value.shape[-1])

    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    # print(query.shape, key.transpose(-2, -1).shape)
    attn = query @ key.transpose(-2, -1)
    attn = attn.softmax(-1)
    # attn = F.dropout(attn, p)
    attn = attn @ value
    return attn.transpose(1, 2)


def move_data(buffer, xk, xv, curr_seq_lens, total_seq):
    batch_size = xk.shape[0]
    num_head = xk.shape[-2]
    head_dim = xk.shape[-1]
    grid = (batch_size, num_head)
    move_data_kernel[grid](
        xk_ptr=xk,
        xv_ptr=xv,
        output_ptr=buffer,
        seq_lens_ptr=curr_seq_lens,
        BATCH_SIZE=batch_size,
        NUM_HEAD=num_head,
        HEAD_DIM=head_dim,
        TOTAL_SEQ=total_seq,
    )


def apply_rotary_pos_emb_triton(q, cos, sin, pos, block_size=128):
    # Get tensor shapes
    batch_size, n_local_heads, head_dim = q.shape

    # Prepare output tensor
    output = torch.empty_like(q)

    # Define grid size
    grid = (batch_size * n_local_heads, head_dim // block_size)

    # Launch kernel
    rotary_embedding_kernel[grid](
        q,
        cos,
        sin,
        output,
        n_local_heads,
        q.stride(0),
        q.stride(1),
        cos.stride(0),
        sin.stride(0),
        output.stride(0),
        output.stride(1),
        pos,
        BLOCK_SIZE=block_size,
    )

    return output
