import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


max_seq_len = 1024
batch_size = 12
seqlen = 512
n_local_heads = 32
n_local_kv_heads = 8
head_dim = 128
freqs_cis = precompute_freqs_cis(
    4096 // 32,
    max_seq_len * 2,
    500000.0,
)

print(freqs_cis.shape)
print(freqs_cis.dtype)

freqs_cis = freqs_cis[0:seqlen]
freqs_cis = freqs_cis.to("cuda")


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if ndim == 4:
        assert freqs_cis.shape == (
            x.shape[1],
            x.shape[-1],
        ), f"{freqs_cis.shape} {x.shape}"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    elif ndim == 3:
        assert freqs_cis.shape == (
            x.shape[0],
            x.shape[-1],
        ), f"{freqs_cis.shape} {x.shape}"
        shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        assert False
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    print(freqs_cis.shape, xq_.shape, xk_.shape, (xq_ * freqs_cis).shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(xq.dim() - 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(xk.dim() - 1)
    return xq_out.type_as(xq), xk_out.type_as(xk)


xq = torch.randn([batch_size, seqlen, n_local_heads, head_dim], device="cuda")
xk = torch.randn([batch_size, seqlen, n_local_kv_heads, head_dim], device="cuda")
apply_rotary_emb(xq, xk, freqs_cis)
