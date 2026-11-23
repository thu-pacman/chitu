# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from omegaconf import OmegaConf

from chitu.device_type import has_native_fp8, is_ascend
from chitu.global_vars import set_global_args, set_quant_variables
from chitu.ops.quant.blockfp8.convert import (
    blockfp8_weight_dequant,
    blockfp8_weight_quant,
)
from chitu.quantization.on_load import (
    convert_state_dict_quantization_on_load,
    validate_quant_on_load_rules,
)
from chitu.quantization.utils import get_quant_from_checkpoint_prefix


def _set_test_global_args(rules=None, soft_fp8=False):
    # `raise_lower_bit_float_to` declares the platform's FP8 mode exactly like the
    # serve config does: float8 on platforms with native FP8 support, bfloat16
    # (soft FP8: FP8 payloads stored as raw uint8, computed by decoding to BF16)
    # everywhere else.
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "raise_lower_bit_float_to": (
                        "bfloat16" if soft_fp8 else "float8_e4m3fn"
                    ),
                    "op_impl": "auto",
                },
                "models": {
                    # Some platform op implementations read `models.name` (e.g. the
                    # NPU soft-FP8 GEMM's dense-model special case), so the minimal
                    # test config carries one.
                    "name": "Test-Model",
                    "quant_config": {"rules": rules or []},
                    "backend_config": {"rules": []},
                },
                "skip_preprocess": False,
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fp8_payload_dtype(soft_fp8: bool) -> torch.dtype:
    # Soft-FP8 platforms store FP8 payloads as raw uint8 (see `Blockfp8Linear`).
    return torch.uint8 if soft_fp8 else torch.float8_e4m3fn


# Tests that run the quantization ops run in both FP8 modes: the native mode
# (float8 payloads, hard-FP8 compute) needs hardware FP8; the soft mode (uint8
# payloads, soft-FP8 compute) is what platforms without FP8 hardware run, so it
# is exercised on every platform.
_BOTH_FP8_MODES = [
    pytest.param(
        False,
        marks=pytest.mark.skipif(
            not has_native_fp8(),
            reason="This test requires the GPU to have native FP8 support",
        ),
    ),
    pytest.param(True),
]

# Tests that request `round_scale_to_pow2=True` require the device's `exp2` to be
# correctly rounded: the quantization computes scales as
# `exp2(ceil(log2(amax / 448)))` and its contract is exact powers of two - UE8M0
# scale storage keeps only the exponent byte of the float, so a scale one ulp off
# decodes as half the scale. Ascend NPU's `exp2` is not correctly rounded (e.g.
# `exp2(-12.0)` is one ulp below `2**-12`), so the quantization raises
# `NotImplementedError` there (see `_blockfp8_weight_quant`); skip these tests.
_requires_exact_pow2_scales = pytest.mark.skipif(
    is_ascend(),
    reason=(
        "Ascend NPU's exp2 is not correctly rounded, so round_scale_to_pow2 "
        "cannot produce exact power-of-two scales and raises NotImplementedError"
    ),
)


def _assert_pow2_scales(scale: torch.Tensor):
    log2_scale = torch.log2(scale.to(torch.float32))
    assert torch.all(log2_scale == torch.round(log2_scale))


def _dequantize_reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # The reference dequantization must run in both CPU-only and GPU test runs, so
    # request the pure-torch implementation instead of the device-dependent "auto".
    return blockfp8_weight_dequant(weight, scale, impl="soft_torch")


def _quantize_blockfp8_ue8m0_1x32(weight: torch.Tensor, soft_fp8: bool):
    """Hand-rolled [1, 32]-block UE8M0 quantization (MiniMax-M3-style MXFP8).

    `blockfp8_weight_quant` supports square scale blocks only, and [1, 32] checkpoints
    come from external tools (vLLM ModelOpt), so build the tensor pair directly.
    """
    blocked = weight.float().view(*weight.shape[:-1], weight.shape[-1] // 32, 32)
    amax = blocked.abs().amax(dim=-1, keepdim=True)
    raw = amax / 448.0
    scale = torch.where(
        raw > 0, torch.exp2(torch.ceil(torch.log2(raw))), torch.ones_like(raw)
    )
    quantized = (blocked / scale).clamp(-448, 448)
    if soft_fp8:
        # Platforms without float8 support cannot run `.to(torch.float8_e4m3fn)` on
        # the device; stock torch supports the dtype on CPU, so encode there and
        # bring the raw payload bytes back.
        payload = (
            quantized.cpu()
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .to(weight.device)
            .view_as(weight)
        )
    else:
        payload = quantized.to(torch.float8_e4m3fn).view_as(weight)
    ue8m0 = (scale.squeeze(-1).contiguous().view(torch.int32) >> 23).to(torch.uint8)
    return payload, ue8m0


def test_set_quant_variables_preserves_from_field():
    cfg = OmegaConf.create(
        {
            "models": {
                "name": "Test-Model",
                "quant_config": {
                    "quant": [
                        {
                            "model": ".",
                            "rules": [
                                {
                                    "regex": "layers\\.",
                                    "from": None,
                                    "type": "blockfp8",
                                },
                                {"regex": "lm_head", "type": "blockfp8"},
                                {
                                    "regex": "experts\\.",
                                    "from": "blockfp8",
                                    "from_kwargs": {"round_scale_to_pow2": False},
                                    "type": "blockfp8",
                                },
                            ],
                        }
                    ]
                },
            }
        }
    )
    set_quant_variables(cfg)

    rules = cfg.models.quant_config.rules
    # An explicit `from: null` survives normalization and means an unquantized source.
    assert "from" in rules[0]
    assert rules[0]["from"] is None
    # An absent `from` stays absent; readers resolve it to the target `type`.
    assert "from" not in rules[1]
    assert cfg.models.quant_config.type == "blockfp8"
    # `from_kwargs` passes normalization through untouched, like `kwargs`.
    assert rules[2]["from_kwargs"] == {"round_scale_to_pow2": False}

    assert (
        get_quant_from_checkpoint_prefix("layers.0.attn.q_proj", rules) == "blockfp8"
    ), "The target quantization resolution must not change"


def test_set_quant_variables_rejects_non_dict_quant_config():
    # `models.quant_config` must be the quant config dict itself. The known footgun is
    # `+models.quant_config=<group-name>`, which assigns the plain group name as a
    # string instead of merging the group; any other non-dict value is rejected by the
    # same check.
    with pytest.raises(ValueError) as error:
        set_quant_variables(
            OmegaConf.create(
                {"models": {"name": "Test-Model", "quant_config": "blockfp8"}}
            )
        )
    assert "must be a dict" in str(error.value)
    assert "defaults" in str(error.value)

    with pytest.raises(ValueError) as error:
        set_quant_variables(
            OmegaConf.create({"models": {"name": "Test-Model", "quant_config": 3}})
        )
    assert "must be a dict" in str(error.value)

    # An absent quant_config is valid: it becomes the default unquantized config.
    cfg = OmegaConf.create({"models": {"name": "Test-Model"}})
    set_quant_variables(cfg)
    assert cfg.models.quant_config.rules == []


@pytest.mark.parametrize(
    "source,target",
    [
        (None, None),
        (None, "mxfp4"),
        ("blockfp8", None),
        ("mxfp4", "blockfp8"),
        ("mxfp4", "mxfp4"),
        ("w8a8_per_token_per_channel_dyn", "blockfp8"),
    ],
)
def test_unimplemented_pairs_raise(source, target):
    rules = [{"regex": "layers\\.", "from": source, "type": target}]
    with pytest.raises(NotImplementedError) as error:
        validate_quant_on_load_rules(rules)
    assert "Not implemented to convert from quantization" in str(error.value)


def test_implemented_pairs_pass_validation():
    validate_quant_on_load_rules(
        [
            {"regex": "layers\\.", "from": None, "type": "blockfp8"},
            {"regex": "lm_head", "type": "blockfp8"},
            {"regex": "wo_a", "from": "blockfp8", "type": "blockfp8"},
            {
                "regex": "experts\\.",
                "from": "blockfp8",
                "from_kwargs": {"round_scale_to_pow2": False},
                "type": "blockfp8",
            },
        ]
    )


def test_from_kwargs_requires_quantized_source():
    # `from_kwargs` describes the detail arguments of a quantized source. An absent
    # `from` means source == target (no conversion) and `from: null` an unquantized
    # source; neither has detail arguments, so the field is rejected instead of
    # silently ignored.
    with pytest.raises(ValueError) as error:
        validate_quant_on_load_rules(
            [{"regex": "layers\\.", "type": "blockfp8", "from_kwargs": {}}]
        )
    assert "`from_kwargs`" in str(error.value)

    with pytest.raises(ValueError) as error:
        validate_quant_on_load_rules(
            [
                {
                    "regex": "layers\\.",
                    "from": None,
                    "from_kwargs": {"round_scale_to_pow2": False},
                    "type": "blockfp8",
                }
            ]
        )
    assert "`from_kwargs`" in str(error.value)


def test_from_kwargs_unknown_keys_raise():
    # A typo'd `from_kwargs` key would be silently ignored by the conversion; reject
    # unknown keys so the claim the conversion uses is exactly what the user wrote.
    with pytest.raises(ValueError) as error:
        validate_quant_on_load_rules(
            [
                {
                    "regex": "layers\\.",
                    "from": "blockfp8",
                    "from_kwargs": {"scale_block_shap": [128, 128]},
                    "type": "blockfp8",
                }
            ]
        )
    assert "scale_block_shap" in str(error.value)


def test_no_from_rules_leave_state_dict_untouched():
    _set_test_global_args()
    state_dict = {"proj.weight": torch.randn(4, 4)}
    rules = [{"regex": "proj", "type": "blockfp8"}]
    assert convert_state_dict_quantization_on_load(state_dict, rules) is state_dict


@_requires_exact_pow2_scales
@pytest.mark.parametrize("soft_fp8", _BOTH_FP8_MODES)
def test_convert_unquantized_to_blockfp8(soft_fp8):
    _set_test_global_args(soft_fp8=soft_fp8)
    device = _device()
    torch.manual_seed(0)
    weight = (
        torch.randn(256, 512, dtype=torch.bfloat16, device=device) * 0.02
    ).contiguous()
    state_dict = {"proj.weight": weight.clone()}
    rules = [
        {
            "regex": "proj",
            "from": None,
            "type": "blockfp8",
            "kwargs": {"round_scale_to_pow2": True},
        }
    ]

    converted = convert_state_dict_quantization_on_load(state_dict, rules)

    assert converted["proj.weight"].dtype == _fp8_payload_dtype(soft_fp8)
    assert converted["proj.weight"].shape == (256, 512)
    assert converted["proj.scale"].dtype == torch.float32
    assert converted["proj.scale"].shape == (2, 4)
    _assert_pow2_scales(converted["proj.scale"])

    dequantized = _dequantize_reference(
        converted["proj.weight"], converted["proj.scale"]
    ).float()
    reference = weight.float()
    assert (dequantized - reference).norm() / reference.norm() < 0.05


@pytest.mark.parametrize("soft_fp8", _BOTH_FP8_MODES)
def test_blockfp8_weight_quant_nd_experts(soft_fp8):
    # Leading dimensions are batch dimensions with their own scales, e.g. stacked
    # per-expert weights.
    _set_test_global_args(soft_fp8=soft_fp8)
    device = _device()
    torch.manual_seed(0)
    weight = (
        torch.randn(4, 256, 512, dtype=torch.bfloat16, device=device) * 0.02
    ).contiguous()

    quantized, scale = blockfp8_weight_quant(weight)

    assert quantized.shape == (4, 256, 512)
    assert quantized.dtype == _fp8_payload_dtype(soft_fp8)
    assert scale.shape == (4, 2, 4)
    dequantized = _dequantize_reference(quantized, scale).float()
    reference = weight.float()
    assert (dequantized - reference).norm() / reference.norm() < 0.05


@pytest.mark.parametrize("soft_fp8", _BOTH_FP8_MODES)
def test_convert_unquantized_per_expert_weights(soft_fp8):
    _set_test_global_args(soft_fp8=soft_fp8)
    device = _device()
    torch.manual_seed(0)
    # MoE checkpoints store one 2-D weight per expert; the expert-stacking merge
    # runs after the conversion, so the pass sees the per-expert keys.
    state_dict = {
        f"experts.{i}.gate_proj.weight": (
            torch.randn(256, 512, dtype=torch.bfloat16, device=device) * 0.02
        ).contiguous()
        for i in range(4)
    }
    rules = [{"regex": "experts", "from": None, "type": "blockfp8"}]

    converted = convert_state_dict_quantization_on_load(state_dict, rules)

    for i in range(4):
        assert converted[f"experts.{i}.gate_proj.weight"].dtype == _fp8_payload_dtype(
            soft_fp8
        )
        assert converted[f"experts.{i}.gate_proj.scale"].shape == (2, 4)


@_requires_exact_pow2_scales
@pytest.mark.parametrize("soft_fp8", _BOTH_FP8_MODES)
def test_convert_blockfp8_requantizes_to_pow2_scales(soft_fp8):
    _set_test_global_args(soft_fp8=soft_fp8)
    device = _device()
    torch.manual_seed(0)
    weight = (
        torch.randn(256, 512, dtype=torch.bfloat16, device=device) * 0.02
    ).contiguous()
    # A blockfp8 checkpoint with plain FP32 (non-pow2) scales.
    quantized, scale = blockfp8_weight_quant(weight)
    state_dict = {"proj.weight": quantized, "proj.scale": scale}
    rules = [
        {
            "regex": "proj",
            "from": "blockfp8",
            "type": "blockfp8",
            "kwargs": {"round_scale_to_pow2": True},
        }
    ]

    converted = convert_state_dict_quantization_on_load(state_dict, rules)

    assert converted["proj.weight"].dtype == _fp8_payload_dtype(soft_fp8)
    _assert_pow2_scales(converted["proj.scale"])
    dequantized = _dequantize_reference(
        converted["proj.weight"], converted["proj.scale"]
    ).float()
    reference = weight.float()
    assert (dequantized - reference).norm() / reference.norm() < 0.05


@_requires_exact_pow2_scales
@pytest.mark.parametrize("soft_fp8", _BOTH_FP8_MODES)
def test_convert_blockfp8_keeps_satisfied_format_untouched(soft_fp8):
    # When the source config (`from` + `from_kwargs`) equals the target config
    # (`type` + `kwargs`), the tensors are kept untouched: they load exactly like a
    # native checkpoint of that format.
    _set_test_global_args(soft_fp8=soft_fp8)
    device = _device()
    torch.manual_seed(0)
    weight = (
        torch.randn(256, 512, dtype=torch.bfloat16, device=device) * 0.02
    ).contiguous()
    rules = [
        {
            "regex": "proj",
            "from": "blockfp8",
            "from_kwargs": {"round_scale_to_pow2": True},
            "type": "blockfp8",
            "kwargs": {"round_scale_to_pow2": True},
        }
    ]

    # FP32-stored pow2 scales, claimed as a pow2 source.
    quantized, scale = blockfp8_weight_quant(weight, round_scale_to_pow2=True)
    state_dict = {"proj.weight": quantized, "proj.scale": scale}
    converted = convert_state_dict_quantization_on_load(state_dict, rules)
    assert converted["proj.weight"] is quantized
    assert converted["proj.scale"] is scale

    # UE8M0 storage with the same semantics on both ends: the storage dtype is read
    # from the tensor and is not part of the config comparison.
    quantized, ue8m0 = _quantize_blockfp8_ue8m0_1x32(
        torch.randn(128, 256, dtype=torch.bfloat16, device=device) * 0.02, soft_fp8
    )
    state_dict = {"proj.weight": quantized, "proj.scale": ue8m0}
    rules_1x32 = [
        {
            "regex": "proj",
            "from": "blockfp8",
            "from_kwargs": {"round_scale_to_pow2": True, "scale_block_shape": [1, 32]},
            "type": "blockfp8",
            "kwargs": {"round_scale_to_pow2": True, "scale_block_shape": [1, 32]},
        }
    ]
    converted = convert_state_dict_quantization_on_load(state_dict, rules_1x32)
    assert converted["proj.weight"] is quantized
    assert converted["proj.scale"] is ue8m0


@_requires_exact_pow2_scales
@pytest.mark.parametrize("soft_fp8", _BOTH_FP8_MODES)
def test_convert_blockfp8_requantizes_unclaimed_pow2_source(soft_fp8):
    # The source format is what the config claims, not what the tensor values happen
    # to be: pow2-valued scales under an unclaimed source config (defaults to
    # non-pow2) differ from the pow2 target, so the group is re-quantized.
    _set_test_global_args(soft_fp8=soft_fp8)
    device = _device()
    torch.manual_seed(0)
    weight = (
        torch.randn(256, 512, dtype=torch.bfloat16, device=device) * 0.02
    ).contiguous()
    quantized, scale = blockfp8_weight_quant(weight, round_scale_to_pow2=True)
    state_dict = {"proj.weight": quantized, "proj.scale": scale}
    rules = [
        {
            "regex": "proj",
            "from": "blockfp8",
            "type": "blockfp8",
            "kwargs": {"round_scale_to_pow2": True},
        }
    ]

    converted = convert_state_dict_quantization_on_load(state_dict, rules)

    assert converted["proj.weight"] is not quantized
    assert converted["proj.scale"] is not scale
    _assert_pow2_scales(converted["proj.scale"])
    dequantized = _dequantize_reference(
        converted["proj.weight"], converted["proj.scale"]
    ).float()
    reference = weight.float()
    assert (dequantized - reference).norm() / reference.norm() < 0.05


@_requires_exact_pow2_scales
@pytest.mark.parametrize("soft_fp8", _BOTH_FP8_MODES)
def test_convert_blockfp8_ue8m0_source_requantizes_to_pow2_blocks(soft_fp8):
    # A [1, 32]-block UE8M0 source claimed with `from_kwargs`: the scale storage
    # dtype comes from the tensor, every other property from the config.
    _set_test_global_args(soft_fp8=soft_fp8)
    device = _device()
    torch.manual_seed(0)
    weight = (
        torch.randn(128, 256, dtype=torch.bfloat16, device=device) * 0.02
    ).contiguous()
    quantized, ue8m0 = _quantize_blockfp8_ue8m0_1x32(weight, soft_fp8)
    state_dict = {"proj.weight": quantized, "proj.scale": ue8m0}
    rules = [
        {
            "regex": "proj",
            "from": "blockfp8",
            "from_kwargs": {"round_scale_to_pow2": True, "scale_block_shape": [1, 32]},
            "type": "blockfp8",
            "kwargs": {"round_scale_to_pow2": True},
        }
    ]

    converted = convert_state_dict_quantization_on_load(state_dict, rules)

    # Target defaults to [128, 128] blocks with FP32-stored pow2 scales.
    assert converted["proj.scale"].dtype == torch.float32
    assert converted["proj.scale"].shape == (1, 2)
    _assert_pow2_scales(converted["proj.scale"])
    dequantized = _dequantize_reference(
        converted["proj.weight"], converted["proj.scale"]
    ).float()
    reference = weight.float()
    assert (dequantized - reference).norm() / reference.norm() < 0.05


@pytest.mark.parametrize("soft_fp8", _BOTH_FP8_MODES)
def test_convert_blockfp8_uses_claimed_scale_block_shape(soft_fp8):
    # The dequantization uses the claimed `from_kwargs` scale block shape.
    _set_test_global_args(soft_fp8=soft_fp8)
    device = _device()
    torch.manual_seed(0)
    weight = (
        torch.randn(256, 512, dtype=torch.bfloat16, device=device) * 0.02
    ).contiguous()
    # A [64, 64]-block source; the target defaults to [128, 128] blocks.
    quantized, scale = blockfp8_weight_quant(weight, scale_block_shape=[64, 64])
    state_dict = {"proj.weight": quantized, "proj.scale": scale}
    rules = [
        {
            "regex": "proj",
            "from": "blockfp8",
            "from_kwargs": {"scale_block_shape": [64, 64]},
            "type": "blockfp8",
        }
    ]

    converted = convert_state_dict_quantization_on_load(state_dict, rules)

    assert converted["proj.scale"].shape == (2, 4)
    dequantized = _dequantize_reference(
        converted["proj.weight"], converted["proj.scale"]
    ).float()
    reference = weight.float()
    assert (dequantized - reference).norm() / reference.norm() < 0.05


def test_convert_blockfp8_wrong_claimed_scale_block_shape_raises():
    # The claimed block shape is validated against the scale tensor, so a false
    # `from_kwargs` claim fails with the claim in the message. The source and target
    # configs compare equal here (both default), so only this validation stands
    # between the false claim and the strict load.
    _set_test_global_args()
    quantized, scale = blockfp8_weight_quant(
        torch.randn(256, 512, dtype=torch.bfloat16), scale_block_shape=[64, 64]
    )
    state_dict = {"proj.weight": quantized, "proj.scale": scale}
    rules = [{"regex": "proj", "from": "blockfp8", "type": "blockfp8"}]

    with pytest.raises(ValueError) as error:
        convert_state_dict_quantization_on_load(state_dict, rules)
    assert "from_kwargs" in str(error.value)
    assert "scale_block_shape" in str(error.value)


def test_convert_blockfp8_rejects_unrenamed_scale_alias():
    # Scale aliases such as `weight_scale_inv` are renamed to the canonical `scale`
    # by the per-model key renames before this pass runs, so the pass only accepts
    # canonical names. A raw alias at this point means the renames did not run.
    _set_test_global_args()
    quantized, scale = blockfp8_weight_quant(
        torch.randn(128, 128, dtype=torch.bfloat16)
    )
    state_dict = {
        "proj.weight": quantized,
        "proj.weight_scale_inv": scale,
    }
    rules = [{"regex": "proj", "from": "blockfp8", "type": "blockfp8"}]

    with pytest.raises(ValueError) as error:
        convert_state_dict_quantization_on_load(state_dict, rules)
    assert "no accompanying scale" in str(error.value)


def test_convert_blockfp8_without_scale_raises():
    # The source and target configs compare equal here (both default), so only the
    # pre-shortcut validation catches the missing scale.
    _set_test_global_args()
    quantized, _ = blockfp8_weight_quant(torch.randn(128, 128, dtype=torch.bfloat16))
    state_dict = {"proj.weight": quantized}
    rules = [{"regex": "proj", "from": "blockfp8", "type": "blockfp8"}]

    with pytest.raises(ValueError) as error:
        convert_state_dict_quantization_on_load(state_dict, rules)
    assert "no accompanying scale" in str(error.value)


def test_convert_false_source_claim_raises():
    # `from` is the user's claim about the checkpoint format, like `type` today. A
    # false claim fails on the structural check of the claimed source, no matter
    # which loader produced the tensors (e.g. a packed-quantized checkpoint under a
    # `from: null` rule).
    _set_test_global_args()
    state_dict = {"proj.weight": torch.zeros(128, 256, dtype=torch.int32)}
    rules = [{"regex": "proj", "from": None, "type": "blockfp8"}]

    with pytest.raises(ValueError) as error:
        convert_state_dict_quantization_on_load(state_dict, rules)
    assert "`from`" in str(error.value)


def test_rule_scoping():
    # Rules are the sole authority on what gets converted: full-name matching, first
    # match wins. A converting rule must scope out layers that stay unquantized
    # (here the MoE router) with a preceding rule without `from`.
    _set_test_global_args()
    state_dict = {
        "layers.0.self_attn.q_proj.weight": torch.randn(32, 16, dtype=torch.bfloat16),
        "layers.0.mlp.gate.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "layers.0.mlp.experts.0.gate_proj.weight": torch.randn(
            32, 16, dtype=torch.bfloat16
        ),
        "layers.0.input_layernorm.weight": torch.randn(16, dtype=torch.bfloat16),
    }
    untouched = dict(state_dict)
    rules = [
        {"regex": r"mlp\.gate\.", "type": None},
        {"regex": r"layers\.", "from": None, "type": "blockfp8"},
    ]

    converted = convert_state_dict_quantization_on_load(state_dict, rules)

    # Everything the converting rule matches is converted, including per-expert
    # tensors, which the merging code stacks afterwards.
    assert converted["layers.0.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn
    assert converted["layers.0.self_attn.q_proj.scale"].shape == (1, 1)
    assert (
        converted["layers.0.mlp.experts.0.gate_proj.weight"].dtype
        == torch.float8_e4m3fn
    )
    assert converted["layers.0.mlp.experts.0.gate_proj.scale"].shape == (1, 1)
    # The router is scoped out by the first rule; 1-D norm weights are never
    # quantizable weights.
    assert (
        converted["layers.0.mlp.gate.weight"] is untouched["layers.0.mlp.gate.weight"]
    )
    assert (
        converted["layers.0.input_layernorm.weight"]
        is untouched["layers.0.input_layernorm.weight"]
    )


@_requires_exact_pow2_scales
@pytest.mark.parametrize("soft_fp8", _BOTH_FP8_MODES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_quantize_on_load_end_to_end_linear(soft_fp8):
    from chitu.native_layout import init_native_layout
    from chitu.quantization.registry import QuantizationRegistry

    rules = [
        {
            "regex": "lm_head",
            "from": None,
            "type": "blockfp8",
            "kwargs": {"round_scale_to_pow2": True},
        }
    ]
    _set_test_global_args(rules, soft_fp8=soft_fp8)
    torch.set_default_dtype(torch.bfloat16)

    cls = QuantizationRegistry.get_quantized_linear_class_from_global_args(
        checkpoint_prefix="lm_head"
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = cls(512, 256, False).cuda()

    model = Model()
    init_native_layout(model)

    torch.manual_seed(0)
    weight = torch.randn(256, 512, device="cuda") * 0.02
    state_dict = {"lm_head.weight": weight.clone()}
    converted = convert_state_dict_quantization_on_load(state_dict, rules)
    # The pass emits exactly the tensor set of a native target-format checkpoint.
    assert converted["lm_head.weight"].dtype == _fp8_payload_dtype(soft_fp8)
    assert converted["lm_head.scale"].dtype == torch.float32

    # Load like the production pipeline does (`assign=True`): native-layout
    # implementations replace parameters with their repacked layout tensors, whose
    # shapes differ from the checkpoint's.
    model.load_state_dict(converted, strict=True, assign=True)

    # The module runs the plain target class. The strict load proves its parameters
    # accept the target format, which always carries a scale while the source format
    # has none. (The parameter dtype itself is not asserted: native-layout classes
    # repack the payload into their own storage.)
    assert model.lm_head.scale is not None

    x = torch.randn(3, 512, device="cuda")
    expected = torch.nn.functional.linear(x, weight).float()
    assert (model.lm_head(x).float() - expected).norm() / expected.norm() < 0.05
