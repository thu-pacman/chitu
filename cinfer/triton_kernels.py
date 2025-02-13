__all__ = ["rotary_embedding_kernel"]


import triton
import triton.language as tl


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
