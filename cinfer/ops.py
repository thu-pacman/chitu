import cinfer_backend, torch


def GEMV(x, w):
    # w = w.transpose(1, 0).contiguous()
    output = torch.zeros(x.shape[:-1] + w.shape[-1:], device=x.device, dtype=x.dtype)
    cinfer_backend.gemv(x, w, output)
    return output


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


import triton
import triton.language as tl


@triton.jit
def move_data_kernel(
    xk_ptr,
    xv_ptr,
    output_ptr,
    seq_lens_ptr,
    BATCH_SIZE: tl.constexpr,
    NUM_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOTAL_SEQ: tl.constexpr,
):
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    dim_id = tl.arange(0, HEAD_DIM)

    xk_offset = batch_id * NUM_HEAD * HEAD_DIM + head_id * HEAD_DIM + dim_id
    xv_offset = xk_offset
    seq_len_offset = batch_id

    xk_data = tl.load(xk_ptr + xk_offset)
    xv_data = tl.load(xv_ptr + xv_offset)
    seq_len = tl.load(seq_lens_ptr + seq_len_offset)

    out_offset_k = (
        batch_id * TOTAL_SEQ * NUM_HEAD * HEAD_DIM
        + seq_len * NUM_HEAD * HEAD_DIM
        + head_id * HEAD_DIM
        + dim_id
    )
    out_offset_v = (BATCH_SIZE * TOTAL_SEQ * NUM_HEAD * HEAD_DIM) + out_offset_k

    tl.store(output_ptr + out_offset_k, xk_data)
    tl.store(output_ptr + out_offset_v, xv_data)


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


@triton.jit
def rotary_embedding_kernel(
    Q,
    COS,
    SIN,
    OUTPUT,
    num_head,
    stride_q1,
    stride_q2,
    stride_cos1,
    stride_sin1,
    stride_out1,
    stride_out2,
    pos_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)

    # Compute batch index and head index
    batch_idx = pid // num_head
    head_idx = pid % num_head

    # Pointers to the beginning of the Q, COS, SIN, and OUTPUT
    Q_ptr = Q + batch_idx * stride_q1 + head_idx * stride_q2
    pos = tl.load(pos_ptr + batch_idx)
    COS_ptr = COS + pos * stride_cos1  # + head_idx * stride_cos1
    SIN_ptr = SIN + pos * stride_sin1  # head_idx * stride_sin1
    OUTPUT_ptr = OUTPUT + batch_idx * stride_out1 + head_idx * stride_out2

    # Create block IDs
    block_id = tl.program_id(axis=1)

    # Create offsets for reading and writing
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offsets_0 = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2)
    offsets_1 = block_id * BLOCK_SIZE + BLOCK_SIZE // 2 + tl.arange(0, BLOCK_SIZE // 2)

    # Load data
    # q = tl.load(Q_ptr + offsets)
    # cos = tl.load(COS_ptr + offsets)
    cos0 = tl.load(COS_ptr + offsets_0)
    cos1 = tl.load(COS_ptr + offsets_1)
    # sin = tl.load(SIN_ptr + offsets)
    sin0 = tl.load(SIN_ptr + offsets_0)
    sin1 = tl.load(SIN_ptr + offsets_1)
    q0 = tl.load(Q_ptr + offsets_0)
    q1 = tl.load(Q_ptr + offsets_1)

    # Rotate half
    # q1 = tl.where(offsets < (offsets.shape[0] // 2), q, -q)
    # q2 = tl.where(offsets >= (offsets.shape[0] // 2), q, -q)
    # q_rotated = tl.cat(q2, q1)

    # Apply rotary embedding
    # q_embed = q * cos + q0 *  # q_rotated * sin
    q_embed0 = q0 * cos0 - q1 * sin0
    q_embed1 = q1 * cos1 + q0 * sin1

    # Store result
    # tl.store(OUTPUT_ptr + offsets, q_embed)
    tl.store(OUTPUT_ptr + offsets_0, q_embed0)
    tl.store(OUTPUT_ptr + offsets_1, q_embed1)


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
