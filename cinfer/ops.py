import cinfer_backend, torch
from .triton_kernels import *


def GEMV(x, w):
    # w = w.transpose(1, 0).contiguous()
    output = torch.zeros(x.shape[:-1] + w.shape[-1:], device=x.device, dtype=x.dtype)
    cinfer_backend.gemv(x, w, output)
    return output


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
