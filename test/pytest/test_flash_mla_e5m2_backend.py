# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

import chitu
import chitu.schemas  # noqa: F401 -- register the Hydra config schema
from chitu.attn_backend.flash_mla_backend import FlashMLABackend
from chitu.device_type import has_accelerator, is_hygon
from chitu.global_vars import set_global_args, set_quant_variables
from chitu.kv_cache.registry import (
    apply_kv_cache_quantization_rules,
    kv_cache_quant_type_for_key,
)
from chitu.models.model_deepseek_v3 import AttentionDeepSeekV3
from chitu.utils import try_import_opt_dep


flash_mla, has_flash_mla = try_import_opt_dep("flash_mla", "flash_mla")


@pytest.mark.skipif(
    not hasattr(torch, "float8_e5m2"),
    reason="torch.float8_e5m2 is unavailable",
)
def test_kimi_k27_e5m2_config_resolves_kv_dtype():
    config_dir = Path(chitu.__file__).resolve().parent / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="serve_config",
            overrides=["models=Kimi-K2.7-Code-kv-e5m2"],
        )
        bf16_cfg = compose(
            config_name="serve_config",
            overrides=["models=Kimi-K2.7-Code"],
        )

    set_quant_variables(cfg)
    set_quant_variables(bf16_cfg)
    kvargs = apply_kv_cache_quantization_rules(
        {"shape_per_token_dict": {"kv_lora_k_pe": (576,)}},
        kv_keys=["kv_lora_k_pe"],
        quant_config=cfg.models.quant_config,
    )

    assert (
        kv_cache_quant_type_for_key(cfg.models.quant_config, "kv_lora_k_pe")
        == "fp8_e5m2"
    )
    assert kvargs["quant_type"] == "fp8_e5m2"
    assert kvargs["dtype_dict"]["kv_lora_k_pe"] == torch.float8_e5m2
    assert (
        kv_cache_quant_type_for_key(bf16_cfg.models.quant_config, "kv_lora_k_pe")
        is None
    )


@pytest.mark.parametrize(
    ("use_e5m2_kv_cache", "is_classic_decoding", "is_decode_stage", "expected"),
    [
        (False, False, False, True),
        (False, True, False, False),
        (True, False, False, True),
        (True, True, False, True),
        (True, True, True, False),
    ],
)
def test_only_e5m2_one_token_prefill_uses_reconstruct_path(
    use_e5m2_kv_cache,
    is_classic_decoding,
    is_decode_stage,
    expected,
):
    attn = object.__new__(AttentionDeepSeekV3)
    attn.mla_absorb = "absorb-kv-only"
    attn.use_e5m2_kv_cache = use_e5m2_kv_cache
    seq_len_delta = SimpleNamespace(
        is_classic_decoding=is_classic_decoding,
        is_decode_stage=is_decode_stage,
    )

    assert attn._should_reconstruct_prefill(seq_len_delta, cp_active=False) is expected
    assert not attn._should_reconstruct_prefill(seq_len_delta, cp_active=True)


@pytest.mark.skipif(
    not has_accelerator() or not is_hygon(),
    reason="requires a Hygon accelerator",
)
def test_dense_e5m2_decode_dispatch_runs_flashmla():
    if not has_flash_mla or not hasattr(
        flash_mla, "flash_mla_with_kvcache_fp8_e5m2_dense"
    ):
        pytest.skip("FlashMLA E5M2 API is unavailable")

    set_global_args(
        OmegaConf.create({"infer": {"op_impl": "torch"}}),
        need_ensure=False,
        need_preprocess=False,
    )

    device = torch.device("cuda")
    batch_size, num_heads, seq_len, page_size = 8, 8, 256, 64
    pages_per_seq = seq_len // page_size
    num_pages = batch_size * pages_per_seq
    latent_dim = 576

    generator = torch.Generator(device=device).manual_seed(20260827)
    choices = torch.tensor(
        [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],
        device=device,
        dtype=torch.bfloat16,
    )
    choice_ids = torch.randint(
        choices.numel(),
        (num_pages, page_size, latent_dim),
        generator=generator,
        device=device,
    )
    cache_e5m2 = choices[choice_ids].to(torch.float8_e5m2).contiguous()
    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(
        batch_size, pages_per_seq
    )

    q_nope = torch.randn(
        (batch_size, num_heads, 512),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    q_pe = torch.randn(
        (batch_size, num_heads, 64),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    new_kv_ids = torch.randint(
        choices.numel(),
        (batch_size, latent_dim),
        generator=generator,
        device=device,
    )
    new_kv = choices[new_kv_ids].contiguous()
    positions = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.int32)
    seq_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
    cache_seqlens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)

    class CacheAccessor:
        def __init__(self):
            self.kv = {"kv_lora_k_pe": cache_e5m2}
            self.block_table = block_table
            self.use_i64_offsets = False

        def get_page_ids(self):
            return self.block_table[seq_ids, positions // page_size]

        def get_offs_in_page(self):
            return positions % page_size

    seq_len_delta = SimpleNamespace(
        is_classic_decoding=True,
        batch_size=batch_size,
        delta_position_ids_tensor_device=positions,
        delta_seq_ids_tensor_device=seq_ids,
        new=SimpleNamespace(lens_tensor_device=cache_seqlens),
    )

    backend = object.__new__(FlashMLABackend)
    backend.use_fp8_cache = False
    backend.use_e5m2_cache = True
    backend.qk_nope_head_dim = 128
    backend.hygon_metadata_decode, _ = flash_mla.get_mla_metadata()

    output = backend.mla_decode_paged_kv(
        q_nope,
        q_pe,
        CacheAccessor(),
        new_kv,
        seq_len_delta,
    )

    q = torch.cat([q_nope, q_pe], dim=-1).view(batch_size, 1, num_heads, latent_dim)
    reference_metadata, _ = flash_mla.get_mla_metadata()
    reference, _ = flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=cache_e5m2.to(torch.bfloat16).unsqueeze(2),
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=512,
        tile_scheduler_metadata=reference_metadata,
        num_splits=None,
        softmax_scale=1.0 / (192**0.5),
        causal=False,
    )
    reference = reference.view(batch_size, num_heads, 512)

    max_abs_diff = (output.float() - reference.float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert output.dtype == torch.bfloat16
    assert max_abs_diff <= 1e-2
    assert cosine == pytest.approx(1.0, abs=1e-6)
