import torch
import uniserve
import torchperf
from uniserve.layers import attention
import torch.nn.functional as F
from diffusers.models.lora import LoRACompatibleConv

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float32)


@torch.no_grad()
def test_RaggedNseqfAttentionForward_uniform():
    n, seq, heads, features = 4, 32, 20, 64
    q = torch.randn(n, seq, heads, features)
    k = torch.randn(n, seq, heads, features)
    v = torch.randn(n, seq, heads, features)
    out0 = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    ).permute(0, 2, 1, 3)
    layer_sdpa = attention.RaggedNseqfAttentionForward()
    idx_cuda, idx_cpu = uniserve.utils.create_index_1d_from_regular(n, seq)
    out1 = layer_sdpa(q.flatten(), k.flatten(), v.flatten(), heads, features, idx_cpu)

    assert torchperf.allclose(out0.flatten(), out1.flatten(), 0.01)


@torch.no_grad()
def test_RaggedNseqfAttentionForward_ragged():
    n, Lseq, heads, features = 4, [32, 16, 8, 4], 20, 64
    q0 = []
    k0 = []
    v0 = []
    out0 = []
    for i, seq in enumerate(Lseq):
        q = torch.randn(seq, heads, features)
        k = torch.randn(seq, heads, features)
        v = torch.randn(seq, heads, features)
        # print(i)
        out = F.scaled_dot_product_attention(
            q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        )
        out = out.permute(1, 0, 2)
        q0.append(q.flatten())
        k0.append(k.flatten())
        v0.append(v.flatten())
        out0.append(out.flatten())

    q0 = torch.concat(q0)
    k0 = torch.concat(k0)
    v0 = torch.concat(v0)
    out0 = torch.concat(out0)

    layer_sdpa = attention.RaggedNseqfAttentionForward()

    idx_cuda, idx_cpu = uniserve.utils.create_index_1d(Lseq)

    out1 = layer_sdpa(
        q0.flatten(), k0.flatten(), v0.flatten(), heads, features, idx_cpu
    )

    assert torchperf.allclose(out0.flatten(), out1.flatten())
