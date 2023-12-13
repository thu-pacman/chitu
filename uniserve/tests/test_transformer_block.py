import pytest
import torch
import torchperf
import uniserve
from diffusers.models.transformer_2d import BasicTransformerBlock, Transformer2DModel
from uniserve.models import RaggedTransformerBlock_nhwc


dtype = torch.float16
torch.set_default_device("cuda")
torch.set_default_dtype(dtype)

# TODO fixed the proper args


def build_transformer(
    attention_bias=True,
    config={
        "dim": 1280,
        "num_attention_heads": 20,
        "attention_head_dim": 64,
        "dropout": 0,
        "cross_attention_dim": 2048,
        "activation_fn": "geglu",
        "num_embeds_ada_norm": 2048,
        "attention_bias": 1280,
        "only_cross_attention": False,
        "double_self_attention": False,
        "upcast_attention": False,
        "norm_elementwise_affine": True,
        "norm_type": "layer_norm",
        "final_dropout": False,
        "attention_type": "default",
    },
):
    config["attention_bias"] = attention_bias
    model = BasicTransformerBlock(**config).eval().cuda().type(dtype)
    return model


@torch.no_grad()
def test_Transformer_uniform():
    m_orig = build_transformer()

    m_ragged = RaggedTransformerBlock_nhwc(m_orig)

    n, seq, heads_num, heads_dim = 2, 64, 20, 64
    hidden_dim = heads_num * heads_dim

    idx1d_cuda, idx1d_cpu = uniserve.utils.create_index_1d([seq] * n)
    cum_idx_cuda = uniserve.utils.create_cum_index_1d([seq] * n)

    x = torch.randn(n, seq, hidden_dim)

    encoder_hidden_states = torch.randn(n, 77, 2048)
    y0 = m_orig(x, encoder_hidden_states=encoder_hidden_states)

    x = x.reshape(-1, hidden_dim)
    y1 = m_ragged(x, cum_idx_cuda, idx1d_cpu, encoder_hidden_states)

    assert torchperf.allclose(y0.flatten(), y1.flatten(), 0.01, 0.01)


@torch.no_grad()
def test_Transformer_ragged():
    n, Lseq, heads_num, heads_dim = 4, [32, 16, 8, 4], 20, 64
    hidden_dim = heads_dim * heads_num
    idx1d_cuda, idx1d_cpu = uniserve.utils.create_index_1d(Lseq)
    cum_idx_cuda = uniserve.utils.create_cum_index_1d(Lseq)

    for bias in [True, False]:
        m_orig = build_transformer(bias)

        m_ragged = RaggedTransformerBlock_nhwc(m_orig)

        encoder_hidden_states = torch.randn(n, 77, 2048)
        x0 = []
        y0 = []
        for i, seq in enumerate(Lseq):
            x = torch.randn(1, seq, hidden_dim)
            y = m_orig(x, encoder_hidden_states=encoder_hidden_states[[i],])
            x0.append(x.flatten())
            y0.append(y.flatten())
        x0 = torch.concat(x0)
        y0 = torch.concat(y0)
        x0 = x0.reshape(-1, hidden_dim)
        y1 = m_ragged(x0, cum_idx_cuda, idx1d_cpu, encoder_hidden_states)

        assert torchperf.allclose(y0.flatten(), y1.flatten(), 0.01)


@pytest.mark.skip("Seqlen (tensor) -> Max length (int) is not supported")
@torch.no_grad()
def test_Transformer_compile():
    n, Lseq, heads_num, heads_dim = 4, [32, 16, 8, 4], 20, 64
    hidden_dim = heads_dim * heads_num
    idx1d_cuda, idx1d_cpu = uniserve.utils.create_index_1d(Lseq)
    cum_idx_cuda = uniserve.utils.create_cum_index_1d(Lseq)

    m_orig = build_transformer()
    m_ragged = RaggedTransformerBlock_nhwc(m_orig)
    encoder_hidden_states = torch.randn(n, 77, 2048)

    x0 = []
    y0 = []

    for i, seq in enumerate(Lseq):
        x = torch.randn(1, seq, hidden_dim)
        y = m_orig(x, encoder_hidden_states=encoder_hidden_states[[i]])
        x0.append(x.flatten())
        y0.append(y.flatten())
    x1 = torch.concat([x.flatten() for x in x0])
    y0 = torch.concat([y.flatten() for y in y0])

    x1 = x1.reshape(-1, hidden_dim)
    y1 = m_ragged(x1, cum_idx_cuda, idx1d_cpu, encoder_hidden_states)
    assert torchperf.allclose(y0.flatten(), y1.flatten(), 0.01)

    # compile wrapper
    m_orig = torch.compile(m_orig, dynamic=True, fullgraph=True)
    m_ragged = torch.compile(m_ragged, dynamic=True, fullgraph=True)

    # Torch dynamo hint
    torch._dynamo.mark_dynamic(x1, 0)
    torch._dynamo.mark_dynamic(encoder_hidden_states, 0)

    y1 = m_ragged(x1, cum_idx_cuda, idx1d_cpu, encoder_hidden_states)
    assert torchperf.allclose(y0.flatten(), y1.flatten(), 0.01)

    def run_orig():
        y0 = []
        for i, x in enumerate(x0):
            y0.append(
                m_orig(
                    x.reshape(1, -1, 1280),
                    encoder_hidden_states=encoder_hidden_states[[i]],
                )
            )
        return y0

    t0 = torchperf.cuda_timeit_ms(run_orig)
    t1 = torchperf.cuda_timeit_ms(
        lambda: m_ragged(x1, cum_idx_cuda, idx1d_cpu, encoder_hidden_states)
    )
    print(f"{t0=} {t1=}")

    # Check recompilation

    n, Lseq, heads_num, heads_dim = 5, [32, 32, 32, 16, 16], 20, 64
    hidden_dim = heads_num * heads_dim

    idx1d_cuda, idx1d_cpu = uniserve.utils.create_index_1d(Lseq)
    cum_idx_cuda = uniserve.utils.create_cum_index_1d(Lseq)
    x0, y0 = [], []
    for i, seq in enumerate(Lseq):
        x = torch.randn(1, seq, hidden_dim)
        x0.append(x)
    x1 = torch.concat([x.flatten() for x in x0])
    x1 = x1.reshape(-1, hidden_dim)
    encoder_hidden_states = torch.randn(n, 77, 2048)
    # m_ragged(x1, n, idx_cpu, features, encoder_hidden_states)
    t2 = torchperf.cuda_timeit_ms(
        lambda: m_ragged(x1, cum_idx_cuda, idx1d_cpu, encoder_hidden_states)
    )
    print(f"{t2=}")
    assert t2 < 10, "An abnormal long execution time hints for recompilation"
