# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Quantization-on-load: convert checkpoint tensors from a source quantization format
to the target (runtime) quantization format while the checkpoint is being loaded.

A quant rule with an explicit ``from`` field requests the conversion (see
``chitu/config/models/quant_config/*.yaml``). Both ends of the conversion are defined
in config: ``from``/``from_kwargs`` describe the source (checkpoint) quantization and
``type``/``kwargs`` the target (runtime) quantization, with omitted kwargs meaning the
same defaults as in the quant module constructors. The source tensors are validated
against the claimed config; a property is read from the tensors only when no config
field expresses it (e.g. the scale storage dtype - blockfp8 checkpoints store pow2
scales either as plain FP32 values or as UE8M0 exponent bytes).

The conversion is a state-dict transform: it rewrites the source-format tensors of one
weight into exactly the tensor set of a native target-format checkpoint, so every
downstream loading step (merging, TP/EP/PP chunking, native layouts, dtype checks) runs
the same code path as if the checkpoint had been quantized offline.

The pass is invoked from the model load pipeline, by
`Transformer.preprocess_state_dict_parallel` between its EP/PP chunking and TP chunking
steps. A subclass that reimplements that pipeline without delegating to `super()` must
invoke `convert_state_dict_quantization_on_load` itself, at the same position: after
EP/PP chunking, so each rank converts only its local tensors, and before TP chunking,
so scales are computed on full-TP weights like a natively quantized checkpoint.
"""

from logging import getLogger
from typing import Any, Optional

import torch
from omegaconf import ListConfig, OmegaConf

from chitu.blockfp8_shape import (
    DEFAULT_ROUND_SCALE_TO_POW2,
    DEFAULT_SCALE_BLOCK_SHAPE,
    blockfp8_scale_dtype,
    blockfp8_scale_shape,
)
from chitu.ops.quant.blockfp8.convert import (
    blockfp8_weight_dequant,
    blockfp8_weight_quant,
)
from chitu.quantization.utils import _match_rule_for_checkpoint_prefix

logger = getLogger(__name__)


# The (source, target) pairs this module implements. Everything else raises
# NotImplementedError, including conversions to a higher-precision quantization: that is
# a fallback implementation of the lower-precision quantization (see
# `infer.raise_lower_bit_float_to`), not a further quantization.
_IMPLEMENTED_PAIRS = {
    (None, "blockfp8"),
    ("blockfp8", "blockfp8"),
}

# Checkpoint tensor names that hold the quantizable weight itself, per source
# quantization. The other tensors of a weight group (scales, bias) are consumed together
# with the weight or pass through untouched.
_SOURCE_WEIGHT_TENSOR_NAMES: dict[Optional[str], tuple[str, ...]] = {
    None: ("weight",),
    "blockfp8": ("weight",),
}

# Canonical checkpoint names of the tensors that hold the scale accompanying a
# `weight`, per source quantization. Raw checkpoint aliases (`weight_scale_inv`,
# `weight_scale`) are renamed to `scale` by the per-model key renames before this
# pass runs - the same canonical names the strict load demands of native checkpoints,
# so accepting aliases here could only mask a pipeline that skipped the renames. An
# unquantized source has no scales.
_SOURCE_SCALE_TENSOR_NAMES: dict[Optional[str], tuple[str, ...]] = {
    "blockfp8": ("scale",),
}

# The kwargs that define the blockfp8 checkpoint format, for `from_kwargs`/`kwargs`
# default resolution and equality. Other kwargs of the blockfp8 modules (e.g.
# `bias_dtype`) are runtime module details, not checkpoint format properties.
_BLOCKFP8_FORMAT_KWARGS = ("scale_block_shape", "round_scale_to_pow2")


def _display_quant_name(quant: Optional[str]) -> str:
    return "none" if quant is None else str(quant)


def _resolved_blockfp8_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve the format-defining blockfp8 kwargs, filling in the defaults.

    The defaults are the blockfp8 module constructor defaults (`Blockfp8Linear` and
    friends), so a rule omitting a kwarg means the same here as it does for module
    construction.
    """
    return {
        "scale_block_shape": list(
            kwargs.get("scale_block_shape", DEFAULT_SCALE_BLOCK_SHAPE)
        ),
        "round_scale_to_pow2": bool(
            kwargs.get("round_scale_to_pow2", DEFAULT_ROUND_SCALE_TO_POW2)
        ),
    }


def validate_quant_on_load_rules(rules: list | ListConfig) -> None:
    """Raise for converting rules whose (source, target) pair is not implemented.

    Called before checkpoint tensors are read so that a misconfigured pair fails within
    seconds of startup instead of after loading gigabytes.
    """
    for rule in rules:
        # `from` is a Python keyword, so the rule field is accessed by key.
        if "from_kwargs" in rule and rule.get("from") is None:
            # An absent `from` means source == target (no conversion) and `from: null`
            # an unquantized source; neither has source detail arguments to describe.
            raise ValueError(
                f"Quant rule (regex: {rule.get('regex')}) sets `from_kwargs`, but its "
                f"source quantization is absent or unquantized: `from_kwargs` "
                f"describes the detail arguments of a quantized source."
            )
        if "from" not in rule:
            continue
        source = rule["from"]
        target = rule.get("type")
        if source == "blockfp8":
            unknown = set(_plain_rule_kwargs(rule, "from_kwargs")) - set(
                _BLOCKFP8_FORMAT_KWARGS
            )
            if unknown:
                raise ValueError(
                    f"Quant rule (regex: {rule.get('regex')}) sets unknown "
                    f"`from_kwargs` keys {sorted(unknown)} for a blockfp8 source; "
                    f"blockfp8 source detail arguments are "
                    f"{list(_BLOCKFP8_FORMAT_KWARGS)}."
                )
        if (source, target) not in _IMPLEMENTED_PAIRS:
            raise NotImplementedError(
                f"Not implemented to convert from quantization "
                f"{_display_quant_name(source)} to quantization "
                f"{_display_quant_name(target)} on load "
                f"(rule regex: {rule.get('regex')})"
            )


def _collect_source_tensors(
    state_dict: dict[str, torch.Tensor], prefix: str, source: Optional[str]
) -> dict[str, torch.Tensor]:
    """Collect one source weight group under `prefix`, by canonical tensor name.

    A missing scale is left for the source validation to report, which names the
    claimed `from` format and the tensor.
    """
    tensors = {"weight": state_dict[prefix + "weight"]}
    for name in _SOURCE_SCALE_TENSOR_NAMES.get(source, ()):
        key = prefix + name
        if key in state_dict:
            tensors[name] = state_dict[key]
    return tensors


def _validate_blockfp8_source_group(
    group: dict[str, torch.Tensor], source_kwargs: dict[str, Any], key: str
) -> None:
    """Validate a weight group against the claimed blockfp8 source config.

    Runs before the same-format shortcut, so a false claim fails with the claim in the
    message on every path - also when the source and target configs compare equal and
    the tensors would otherwise flow to the strict load, whose errors point at the
    model side instead of the config.
    """
    scale = group.get("scale")
    if scale is None:
        raise ValueError(
            f"Quantization-on-load rule claims a blockfp8 (`from: blockfp8`) source, "
            f"but checkpoint tensor {key} has no accompanying scale tensor. Check "
            f"that `from` matches the actual checkpoint format."
        )
    weight = group["weight"]
    if weight.element_size() != 1:
        raise ValueError(
            f"Quantization-on-load rule claims a blockfp8 (`from: blockfp8`) source, "
            f"but checkpoint tensor {key} uses {weight.dtype} storage instead of "
            f"one-byte FP8 storage. Check that `from` matches the actual checkpoint "
            f"format."
        )
    scale_block_shape = _resolved_blockfp8_kwargs(source_kwargs)["scale_block_shape"]
    expected_scale_shape = blockfp8_scale_shape(
        weight.shape[-2], weight.shape[-1], scale_block_shape=scale_block_shape
    )
    if tuple(scale.shape[-2:]) != tuple(expected_scale_shape):
        raise ValueError(
            f"Quantization-on-load rule claims a blockfp8 source with "
            f"scale_block_shape {scale_block_shape} (`from_kwargs`), but checkpoint "
            f"tensor {key} has a scale of shape {tuple(scale.shape[-2:])}, expected "
            f"{tuple(expected_scale_shape)}. Check that `from_kwargs` matches the "
            f"actual checkpoint format."
        )


def _decode_ue8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    # UE8M0 scales are stored as exponent bytes (biased by 127), either as uint8 or as
    # the raw float8_e8m0fnu dtype. `_handle_quantized_weights_casting` already decodes
    # the float8_e8m0fnu variant when the target scale dtype is FP32; decode here too so
    # the dequantization below works no matter which form reached this point.
    exponent = scale.view(torch.uint8).to(torch.float32)
    return torch.exp2(exponent - 127.0)


def _dequantize_blockfp8_source(
    group: dict[str, torch.Tensor], source_kwargs: dict[str, Any], key: str
) -> torch.Tensor:
    # Called after `_validate_blockfp8_source_group` checked the group against the
    # claimed config.
    weight, scale = group["weight"], group["scale"]
    scale_block_shape = _resolved_blockfp8_kwargs(source_kwargs)["scale_block_shape"]
    # The scale storage is the one source property no config field expresses: pow2
    # scales are stored either as plain FP32 values or as UE8M0 exponent bytes, so
    # read it from the tensor dtype.
    if scale.dtype == torch.uint8 or scale.dtype == getattr(
        torch, "float8_e8m0fnu", None
    ):
        scale = _decode_ue8m0_scales(scale)
    # The conversion runs before the state dict is prefetched to the GPU, so the
    # tensors are usually still on CPU, where the triton implementations cannot run:
    # request the pure-torch implementation instead of "auto". It decodes both float8
    # and uint8-viewed storage, and is exact for every finite E4M3 payload.
    return blockfp8_weight_dequant(
        weight, scale, scale_block_shape=scale_block_shape, impl="soft_torch"
    )


def _dequantize_source(
    group: dict[str, torch.Tensor],
    source: Optional[str],
    source_kwargs: dict[str, Any],
    key: str,
) -> torch.Tensor:
    if source is None:
        weight = group["weight"]
        if not weight.is_floating_point():
            raise ValueError(
                f"Quantization-on-load rule claims an unquantized (`from: null`) "
                f"source, but checkpoint tensor {key} has non-floating dtype "
                f"{weight.dtype}. Check that `from` matches the actual checkpoint "
                f"format."
            )
        return weight
    if source == "blockfp8":
        return _dequantize_blockfp8_source(group, source_kwargs, key)
    raise NotImplementedError(
        f"Not implemented to convert from quantization {_display_quant_name(source)} "
        f"on load"
    )


def _blockfp8_source_satisfies_target(
    source_kwargs: dict[str, Any], target_kwargs: dict[str, Any]
) -> bool:
    """Return whether a source blockfp8 quant config is identical to the target config."""

    return _resolved_blockfp8_kwargs(source_kwargs) == _resolved_blockfp8_kwargs(
        target_kwargs
    )


def _quantize_to_blockfp8(
    weight: torch.Tensor, target_kwargs: dict[str, Any]
) -> dict[str, torch.Tensor]:
    resolved = _resolved_blockfp8_kwargs(target_kwargs)
    # The dispatcher's "auto" resolves by `infer.raise_lower_bit_float_to`, so the
    # payload storage already matches the target module parameters: float8 where
    # the platform runs FP8 compute, raw uint8 where it does not. This matters
    # because `_handle_quantized_weights_casting` has already run at this point
    # and will not post-process the converted tensors.
    quantized, scale = blockfp8_weight_quant(
        weight,
        scale_block_shape=resolved["scale_block_shape"],
        round_scale_to_pow2=resolved["round_scale_to_pow2"],
    )
    scale_dtype = blockfp8_scale_dtype(
        resolved["round_scale_to_pow2"], resolved["scale_block_shape"]
    )
    if scale_dtype == torch.uint8:
        # UE8M0 storage: keep only the exponent byte of the (pow2-valued) scale.
        scale = (scale.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    return {"weight": quantized, "scale": scale}


def _quantize_to_target(
    weight: torch.Tensor,
    target: Optional[str],
    target_kwargs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if target == "blockfp8":
        return _quantize_to_blockfp8(weight, target_kwargs)
    raise NotImplementedError(
        f"Not implemented to convert to quantization {_display_quant_name(target)} "
        f"on load"
    )


def _convert_weight_group(
    group: dict[str, torch.Tensor],
    source: Optional[str],
    source_kwargs: dict[str, Any],
    target: Optional[str],
    target_kwargs: dict[str, Any],
    key: str,
) -> Optional[dict[str, torch.Tensor]]:
    """Convert one source weight group into target-format tensors.

    Returns None when the source and target configs describe the same format and the
    group should be kept untouched.
    """
    if source == "blockfp8":
        _validate_blockfp8_source_group(group, source_kwargs, key)
        if target == "blockfp8" and _blockfp8_source_satisfies_target(
            source_kwargs, target_kwargs
        ):
            return None
    high_precision = _dequantize_source(group, source, source_kwargs, key)
    return _quantize_to_target(high_precision, target, target_kwargs)


def _plain_rule_kwargs(rule, field: str) -> dict[str, Any]:
    kwargs = rule.get(field) or {}
    if OmegaConf.is_config(kwargs):
        # Plain Python containers, not `ListConfig`/`DictConfig`, matching what
        # `get_quant_kwargs_from_checkpoint_prefix` produces.
        kwargs = OmegaConf.to_container(kwargs, resolve=True)
    return dict(kwargs)


def convert_state_dict_quantization_on_load(
    state_dict: dict[str, torch.Tensor],
    rules: list | ListConfig,
) -> dict[str, torch.Tensor]:
    """Rewrite source-format checkpoint tensors into target-format tensors, in place.

    Only rules with an explicit `from` field convert anything; a state dict under rules
    without `from` is returned untouched. See the module docstring for the mechanism.

    The rules are the sole authority on what gets converted. They are matched on the
    full tensor names, first match wins - the same resolution that initializes the
    model's modules - so the converted tensors match the model by construction. A
    converting rule must therefore scope precisely what it converts: a layer that
    stays unquantized (e.g. the MoE router) must be matched by an earlier rule
    without `from`, or its converted tensors fail the strict load.
    """
    if not any("from" in rule or "from_kwargs" in rule for rule in rules):
        return state_dict
    validate_quant_on_load_rules(rules)

    converted_groups = 0
    for key in list(state_dict.keys()):
        rule = _match_rule_for_checkpoint_prefix(key, rules)
        if rule is None or "from" not in rule:
            continue

        source = rule["from"]
        source_kwargs = _plain_rule_kwargs(rule, "from_kwargs")
        target = rule.get("type")
        tensor_name = key.rsplit(".", 1)[-1]
        if tensor_name not in _SOURCE_WEIGHT_TENSOR_NAMES.get(source, ()):
            continue
        if state_dict[key].dim() < 2:
            continue  # 1-D "weight" tensors (norms) are never quantized weights.

        module_path = key[: -len(tensor_name) - 1]
        prefix = module_path + "." if module_path else ""
        group = _collect_source_tensors(state_dict, prefix, source)
        new_tensors = _convert_weight_group(
            group,
            source,
            source_kwargs,
            target,
            _plain_rule_kwargs(rule, "kwargs"),
            key,
        )
        if new_tensors is None:
            continue
        for name in group:
            state_dict.pop(prefix + name)
        for name, tensor in new_tensors.items():
            state_dict[prefix + name] = tensor
        converted_groups += 1

    if converted_groups > 0:
        logger.debug(
            f"Quantization-on-load converted {converted_groups} weight groups in this "
            f"state-dict part"
        )
    return state_dict
