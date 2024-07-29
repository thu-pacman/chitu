import torch
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


# Example usage
max_seq_len = 1024
head_dim = 128
batch_size = 16
n_local_heads = 8

# Sample data
q = torch.randn(batch_size, n_local_heads, head_dim, device="cuda")
cos = torch.randn(max_seq_len, head_dim, device="cuda")
sin = torch.randn(max_seq_len, head_dim, device="cuda")
position_ids = torch.randint(0, max_seq_len, (batch_size,), device="cuda")

# Apply rotary position embedding using Triton
q_embed = apply_rotary_pos_emb_triton(q, cos, sin, position_ids)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


cos = cos[position_ids].unsqueeze(1)
sin = sin[position_ids].unsqueeze(1)
q_embed2 = (q * cos) + (rotate_half(q) * sin)

print(q_embed.shape)
print(q_embed)
print(q_embed2)
print(torch.sum(torch.isclose(q_embed, q_embed2)) // q_embed.numel())
