from xformers.ops import fmha
import torch
import torch.nn.functional as F
import timeit


@torch.inference_mode()
def ground_truth_attention(query, key, value):
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


@torch.inference_mode()
def test_decode_attention(query, key, value):
    if key.shape[-2] < query.shape[-2] and key.shape[-2] == 1:
        new_shape = []
        for i in range(len(key.shape)):
            if i == len(key.shape) - 2:
                new_shape.append(query.shape[-2])
            else:
                new_shape.append(key.shape[i])
        key = key.expand(new_shape)
        value = value.expand(new_shape)
    output = fmha.memory_efficient_attention_forward(xq, key, value)
    return output


def prepare_data(batch=1, seq=10, seq_k=1000, n_local_kv_heads=32, head_dim=128):
    xq = torch.randn(
        [batch, seq, n_local_kv_heads, head_dim],
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        [batch, seq_k, n_local_kv_heads, head_dim],
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        [batch, seq_k, n_local_kv_heads, head_dim],
        device="cuda",
        dtype=torch.bfloat16,
    )
    return xq, key, value


# prepare data for group attention
def prepare_data_gqa(
    batch=1, seq=10, seq_k=1000, n_local_kv_heads=8, head_dim=128, heads_per_group=4
):
    xq = torch.randn(
        [batch, seq, n_local_kv_heads, heads_per_group, head_dim],
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        [batch, seq_k, n_local_kv_heads, 1, head_dim],
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        [batch, seq_k, n_local_kv_heads, 1, head_dim],
        device="cuda",
        dtype=torch.bfloat16,
    )
    return xq, key, value


def prepare_noncontinuous_data(
    batch=1, seq=10, seq_k=1000, n_local_kv_heads=32, head_dim=128
):
    xq = torch.randn(
        [batch, seq, n_local_kv_heads, head_dim],
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        [batch, seq_k, n_local_kv_heads, head_dim],
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        [batch, seq_k, n_local_kv_heads, head_dim],
        device="cuda",
        dtype=torch.bfloat16,
    )
    return xq, key, value


if __name__ == "__main__":
    # xq, key, value = prepare_data()
    xq, key, value = prepare_data_gqa()
    o1 = test_decode_attention(xq, key, value)
    o2 = ground_truth_attention(xq, key, value)
    print(o1.shape, o2.shape)
    o1 = o1.reshape(o2.shape)
    print(torch.isclose(o1, o2).sum() / o1.numel())
    print(
        "xformer",
        timeit.timeit(lambda: test_decode_attention(xq, key, value), number=100) / 100,
    )
    print(
        "ground_truth",
        timeit.timeit(lambda: ground_truth_attention(xq, key, value), number=100) / 100,
    )
