import pytest
import torch
import torchperf
import uniserve
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from uniserve.models import RaggedTransformer2DModel_nchw
import torch._dynamo

torch._dynamo.config.suppress_errors = True

dtype = torch.float16
torch.set_default_device("cuda")
torch.set_default_dtype(dtype)

torch.manual_seed(242)


def build_transformer2d(
    config={
        "num_attention_heads": 10,
        "attention_head_dim": 64,
        "in_channels": 640,
        "out_channels": None,
        "num_layers": 2,
        "dropout": 0,
        "norm_num_groups": 32,
        "cross_attention_dim": 2048,
        "attention_bias": False,
        "sample_size": None,
        "num_vector_embeds": None,
        "patch_size": None,
        "activation_fn": "geglu",
        "num_embeds_ada_norm": None,
        "use_linear_projection": True,
        "only_cross_attention": False,
        "double_self_attention": False,
        "upcast_attention": None,
        "norm_type": "layer_norm",
        "norm_elementwise_affine": True,
        "attention_type": "default",
    }
):
    model = Transformer2DModel(**config).eval().cuda().type(dtype)
    return model


@torch.no_grad()
def test_Transformer2d_uniform():
    m_orig = build_transformer2d()

    m_ragged = RaggedTransformer2DModel_nchw(m_orig)

    n, heads_num, heads_dim, h, w = 2, 10, 64, 32, 32

    c = heads_num * heads_dim

    idx_cuda, idx_cpu = uniserve.utils.create_index_2d_from_regular(n, h, w)
    cum_idx_cuda = uniserve.utils.create_cum_index_1d([h * w] * n)

    x = torch.randn(n, c, h, w)
    encoder_hidden_states = torch.randn(n, 77, 2048)

    y0 = m_orig(hidden_states=x, encoder_hidden_states=encoder_hidden_states)
    y1 = m_ragged(
        x.flatten(),
        heads_num,
        heads_dim,
        c,
        idx_cuda,
        idx_cpu,
        cum_idx_cuda,
        encoder_hidden_states,
    )
    assert torchperf.allclose(y0.sample.flatten(), y1, 0.01)


@torch.no_grad()
def test_Transformer2d_ragged():
    n, heads_num, heads_dim, hs, ws = 4, 10, 64, [14, 14, 28, 28], [14, 28, 14, 28]
    c = heads_num * heads_dim
    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)
    cum_idx_cuda = uniserve.utils.create_cum_index_1d([h * w for h, w in zip(hs, ws)])

    m_orig = build_transformer2d()
    m_ragged = RaggedTransformer2DModel_nchw(m_orig)

    encoder_hidden_states = torch.randn(n, 77, 2048)

    x0 = []
    y0 = []
    for i, (h, w) in enumerate(zip(hs, ws)):
        x = torch.randn(1, c, h, w)
        y = m_orig(hidden_states=x, encoder_hidden_states=encoder_hidden_states[[i]])
        x0.append(x.flatten())
        y0.append(y.sample.flatten())
    x0 = torch.concat(x0)
    y0 = torch.concat(y0)

    y1 = m_ragged(
        x0.flatten(),
        heads_num,
        heads_dim,
        c,
        idx_cuda,
        idx_cpu,
        cum_idx_cuda,
        encoder_hidden_states,
    )

    assert torchperf.allclose(y0.flatten(), y1.flatten(), 0.01)


@pytest.mark.skip("Dynamo fails on Transformer_block")
@torch.no_grad()
def test_RaggedTransformer2d_compile():
    n, heads_num, heads_dim, hs, ws = (
        6,
        10,
        64,
        [14, 14, 28, 28, 16, 32],
        [23, 34, 14, 28, 14, 28],
    )
    c = heads_num * heads_dim
    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)
    cum_idx_cuda = uniserve.utils.create_cum_index_1d([h * w for h, w in zip(hs, ws)])

    m_orig = build_transformer2d()
    m_ragged = RaggedTransformer2DModel_nchw(m_orig)

    encoder_hidden_states = torch.randn(n, 77, 2048)

    x0 = []
    y0 = []
    for i, (h, w) in enumerate(zip(hs, ws)):
        x = torch.randn(1, c, h, w)
        y = m_orig(hidden_states=x, encoder_hidden_states=encoder_hidden_states[[i]])
        x0.append(x)
        y0.append(y.sample)
    x1 = torch.concat([x.flatten() for x in x0])
    y0 = torch.concat([y.flatten() for y in y0])

    y1 = m_ragged(
        x1.flatten(),
        heads_num,
        heads_dim,
        c,
        idx_cuda,
        idx_cpu,
        cum_idx_cuda,
        encoder_hidden_states,
    )
    assert torchperf.allclose(y0.flatten(), y1.flatten(), 0.1)

    m_orig = torch.compile(m_orig, dynamic=True, fullgraph=True)
    m_ragged = torch.compile(m_ragged, dynamic=True, fullgraph=True)

    torch._dynamo.mark_dynamic(x1, 0)
    torch._dynamo.mark_dynamic(encoder_hidden_states, 0)
    torch._dynamo.mark_dynamic(idx_cuda, 1)
    torch._dynamo.mark_dynamic(idx_cpu, 1)

    y1 = m_ragged(
        x1.flatten(),
        heads_num,
        heads_dim,
        c,
        idx_cuda,
        idx_cpu,
        cum_idx_cuda,
        encoder_hidden_states,
    )
    assert torchperf.allclose(y0.flatten(), y1.flatten(), 0.1)

    def run_orig():
        y0 = []
        for i, x in enumerate(x0):
            y0.append(
                m_orig(
                    hidden_states=x, encoder_hidden_states=encoder_hidden_states[[i]]
                ).sample
            )
        return y0

    # run_orig results in torch._dynamo.exc.Unsupported since it returns Transformer2DModelOutput
    t0 = 0.0  # torchperf.cuda_timeit_ms(run_orig)
    t1 = torchperf.cuda_timeit_ms(
        lambda: m_ragged(
            x1,
            heads_num,
            heads_dim,
            c,
            idx_cuda,
            idx_cpu,
            cum_idx_cuda,
            encoder_hidden_states,
        )
    )
    print(f"{t0=} {t1=}")

    n, heads_num, heads_dim, hs, ws = (
        5,
        10,
        64,
        [14, 14, 28, 28, 20],
        [14, 28, 14, 28, 20],
    )
    c = heads_num * heads_dim
    idx_cuda, idx_cpu = uniserve.utils.create_index_2d(hs, ws)

    x0, y0 = [], []

    for i, (h, w) in enumerate(zip(hs, ws)):
        x = torch.randn(1, c, h, w)
        x0.append(x)

    x1 = torch.concat([x.flatten() for x in x0])
    encoder_hidden_states = torch.randn(n, 77, 2048)
    t2 = torchperf.cuda_timeit_ms(
        lambda: m_ragged(
            x1,
            heads_num,
            heads_dim,
            c,
            idx_cuda,
            idx_cpu,
            cum_idx_cuda,
            encoder_hidden_states,
        )
    )
    print(f"{t2=}")
    assert t2 < 10, "An abnormal long execution time hints for recompilation."
