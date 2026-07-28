# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the DSA indexer Q/K builder."""

import math
import einops
import pytest
import torch
from omegaconf import OmegaConf

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.global_vars import set_global_args, get_global_args
from chitu.ops import apply_rotary_pos_emb_partial, hadamard_transform
from chitu.testing import assert_close
from chitu.utils import try_import_opt_dep
from chitu.utils import try_import_and_setup_torch_npu
from chitu.device_type import has_native_fp8

scipy, has_scipy = try_import_opt_dep("scipy", "scipy")
fast_hadamard_transform, has_fast_hadamard_transform = try_import_opt_dep(
    "fast_hadamard_transform", "fast_hadamard_transform"
)
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


class MonkIndexerImpl:
    """Monkey DSAIndexer impl."""

    def __init__(self, impl: str):
        self.impl = impl


def _make_indexer(
    n_heads,
    head_dim,
    rope_head_dim,
    impl,
    rope_layout,
    device,
    *,
    fp8_indexer_kv=False,
):
    set_global_args(
        OmegaConf.create(
            {
                "infer": {"max_seq_len": 4096, "op_impl": "torch"},
                "models": {
                    "index_n_heads": n_heads,
                    "index_head_dim": head_dim,
                    "dim": n_heads * head_dim,
                    "qk_rope_head_dim": rope_head_dim,
                    "q_lora_rank": 0,
                    "index_topk": 2048,
                    "index_rope_layout": rope_layout,
                    "index_norm_dtype": "float32",
                    "quant_config": {
                        "kv_cache": {
                            "rules": (
                                [
                                    {
                                        "regex": "^indexer_k$",
                                        "type": "fp8_pertoken_indexer",
                                    }
                                ]
                                if fp8_indexer_kv
                                else []
                            )
                        }
                    },
                },
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    from chitu.models.model_deepseek_v3 import Indexer

    model_cfg = get_global_args().models
    indexer = Indexer(
        model_cfg,
        checkpoint_prefix="indexer",
        indexer_impl=MonkIndexerImpl(impl),
    )
    return indexer.to(device)


def _make_freqs_cis(s, rope_head_dim, device):
    complex_freqs = torch.polar(
        torch.ones(s, rope_head_dim // 2, device=device, dtype=torch.float32),
        torch.rand(s, rope_head_dim // 2, device=device, dtype=torch.float32)
        * 2
        * math.pi,
    )
    return BatchedFreqsCis(
        complex_freqs.real.contiguous().to(torch.bfloat16),
        complex_freqs.imag.contiguous().to(torch.bfloat16),
    )


def _dequant_blockfp8(x_fp8, scale, block_size=128):
    x = x_fp8.to(torch.float32)
    x_blocked = x.view(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    dequant = x_blocked * scale.unsqueeze(-1)
    return dequant.view(x_fp8.shape)


def _ref_qk_transform(
    indexer,
    q,
    k,
    freqs_cis,
    head_dim,
    rope_head_dim,
    rope_layout,
    *,
    use_hadamard_transform,
):
    """Reference"""
    q3 = einops.rearrange(q.clone(), "s (h d) -> s h d", d=head_dim)
    k_normed = indexer.k_norm(k.clone())
    q_rot, k_rot, _, _, _, _, _, _ = apply_rotary_pos_emb_partial(
        q3,
        k_normed,
        freqs_cis,
        q_rotary_end=rope_head_dim,
        k_rotary_end=rope_head_dim,
        rotary_type=rope_layout,
        impl="torch_npu" if has_torch_npu else "auto",
    )
    if use_hadamard_transform:
        q_rot = hadamard_transform(q_rot, scale=head_dim**-0.5)
        k_rot = hadamard_transform(k_rot, scale=head_dim**-0.5)
    return q_rot, k_rot


@pytest.mark.parametrize("impl", ["torch_bf16", "hygon"])
@pytest.mark.parametrize("rope_layout", ["separated", "interleaved"])
@pytest.mark.parametrize("s", [1, 8])
def test_build_index_qk_bf16_path(impl, rope_layout, s):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    device = "cuda"

    n_heads, head_dim, rope_head_dim = 4, 128, 64
    indexer = _make_indexer(n_heads, head_dim, rope_head_dim, impl, rope_layout, device)

    x = torch.randn(s, n_heads * head_dim, device=device)
    q = torch.randn(s, n_heads * head_dim, device=device)
    k = torch.randn(s, head_dim, device=device)
    freqs_cis = _make_freqs_cis(s, rope_head_dim, device)

    (q_out, q_scale), (k_out, k_scale) = indexer._build_index_qk(
        x, q.clone(), k.clone(), freqs_cis
    )
    # The bf16/hygon branch returns plain tensors with a None scale.
    assert q_scale is None and k_scale is None
    assert q_out.shape == (s, n_heads, head_dim)
    assert k_out.shape == (s, head_dim)

    q_ref, k_ref = _ref_qk_transform(
        indexer,
        q,
        k,
        freqs_cis,
        head_dim,
        rope_head_dim,
        rope_layout,
        use_hadamard_transform=False,
    )

    assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not has_scipy and not has_fast_hadamard_transform,
    reason="A Hadamard transform implementation (scipy or fast_hadamard_transform) is required;",
)
@pytest.mark.skipif(not has_native_fp8(), reason="Float8_e4m3fn support is required;")
@pytest.mark.parametrize("impl", ["deepgemm", "triton"])
@pytest.mark.parametrize("rope_layout", ["separated", "interleaved"])
@pytest.mark.parametrize("s", [1, 8])
def test_build_index_qk_fp8_path(impl, rope_layout, s):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    device = "cuda"

    n_heads, head_dim, rope_head_dim = 4, 128, 64
    indexer = _make_indexer(
        n_heads,
        head_dim,
        rope_head_dim,
        impl,
        rope_layout,
        device,
        fp8_indexer_kv=True,
    )

    x = torch.randn(s, n_heads * head_dim, device=device)
    q = torch.randn(s, n_heads * head_dim, device=device)
    k = torch.randn(s, head_dim, device=device)
    freqs_cis = _make_freqs_cis(s, rope_head_dim, device)

    (q_fp8, q_scale), (k_fp8, k_scale) = indexer._build_index_qk(
        x, q.clone(), k.clone(), freqs_cis
    )
    # The fp8 path returns quantized tensors with a per-block scale.
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert q_scale is not None and k_scale is not None
    assert q_fp8.shape == (s, n_heads, head_dim)
    assert k_fp8.shape == (s, head_dim)

    q_out = _dequant_blockfp8(q_fp8, q_scale, block_size=indexer.block_size)
    k_out = _dequant_blockfp8(k_fp8, k_scale, block_size=indexer.block_size)

    q_ref, k_ref = _ref_qk_transform(
        indexer,
        q,
        k,
        freqs_cis,
        head_dim,
        rope_head_dim,
        rope_layout,
        use_hadamard_transform=True,
    )

    assert_close(q_out, q_ref.float(), rtol=5e-2, atol=1e-1, cos_sim_tol=1e-2)
    assert_close(k_out, k_ref.float(), rtol=5e-2, atol=1e-1, cos_sim_tol=1e-2)
