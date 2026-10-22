# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from contextlib import suppress

from chitu.global_vars import get_global_args

from .type_def import OutputParserParams, OutputParserType

logger = logging.getLogger(__name__)


@functools.cache
def get_output_parser_marker_token_ids():
    """Derive the classic single-tag reasoning marker ids from the tokenizer.

    Encodes ``"<think></think>"`` and expects exactly the two marker tokens
    (both single added tokens for classic reasoning models). For models
    whose tokenizer encodes the string as ordinary text instead (the
    two-way unpack raises) the derivation fails; their markers must come
    from config (``models.reasoning_start_delimiter_ids`` /
    ``models.reasoning_end_delimiter_ids``).
    """
    from chitu.backend import Backend

    if Backend.tokenizer is None:
        logger.warning(
            "get output parser marker token ids before tokenizer initialized, "
            "only make sense in unit test"
        )
        return -1, -1

    with suppress(Exception):
        reasoning_start_token_id, reasoning_end_token_id = Backend.tokenizer.encode(
            "<think></think>", bos=False, eos=False
        )
        return reasoning_start_token_id, reasoning_end_token_id

    logger.info("This model does not support reasoning")
    return -1, -1


def _configured_delimiter(name: str) -> list[int] | None:
    """Read a delimiter token-id list from the model config; None if unset.

    A bare int is accepted and treated as the length-1 special case of the
    list form (``id`` means ``[id]``). A configured-but-invalid value (empty
    or non-int entries) raises: a wrong delimiter would silently mis-parse
    every request of the model. A missing global config or a config without
    a ``models`` section (unit-test contexts) counts as unset.
    """
    try:
        args = get_global_args()
        ids = getattr(args.models, name, None)
    except Exception:
        return None
    if ids is None:
        return None
    if isinstance(ids, int):
        return [ids]
    try:
        ids = list(ids)
    except TypeError:
        ids = None
    if not ids or not all(isinstance(t, int) for t in ids):
        raise ValueError(
            f"models.{name} must be an int token id or a non-empty list of "
            f"int token ids, got {ids!r}"
        )
    return ids


def _resolve_reasoning_delimiter_ids() -> tuple[list[int] | None, list[int] | None]:
    """Resolve the reasoning start/end delimiters, independently per field.

    Delimiter resolution is ORTHOGONAL to the reasoning type: any type may
    use configured delimiters or the tokenizer-derived classic markers, and
    a single-token delimiter is just the length-1 special case of a
    multi-token one. Configured ids (``models.reasoning_start_delimiter_ids``
    / ``models.reasoning_end_delimiter_ids``) win per field; a field without
    config falls back to the classic single-tag markers derived from the
    tokenizer. Returns ``(start_ids, end_ids)``, each None when unavailable.
    """
    start_ids = _configured_delimiter("reasoning_start_delimiter_ids")
    end_ids = _configured_delimiter("reasoning_end_delimiter_ids")
    if start_ids is None or end_ids is None:
        start_id, end_id = get_output_parser_marker_token_ids()
        if start_ids is None and start_id != -1:
            start_ids = [start_id]
        if end_ids is None and end_id != -1:
            end_ids = [end_id]
    return start_ids, end_ids


def _get_output_parser_params(enable_thinking: bool) -> OutputParserParams:
    try:
        args = get_global_args()
        parser_type: OutputParserType = args.models.reasoning_type
    except Exception:
        # Model not configured with a reasoning_type: fall back to "auto".
        logger.info('fallback to "auto" output parser type')
        parser_type = "auto"

    reasoning_start_ids, reasoning_end_ids = _resolve_reasoning_delimiter_ids()
    output_end_ids = _configured_delimiter("output_end_delimiter_ids")

    if parser_type == "auto":
        # Reasoning-capable iff both reasoning markers are available.
        parser_type = (
            "triggered" if reasoning_start_ids and reasoning_end_ids else "disabled"
        )

    if parser_type == "switchable":
        # Reasoning can be turned off per request; off requires the chat
        # template to render a non-thinking prompt (the serve layer passes
        # the request's setting under the kwarg name the template reads).
        parser_type = "forced" if enable_thinking else "disabled"

    if parser_type == "triggered" and not enable_thinking:
        parser_type = "disabled"

    if parser_type == "disabled":
        # No reasoning channel — but the sequence-ending marker still applies
        # (e.g. Kimi-K3 with thinking off still ends its message with the
        # multi-token XTML marker).
        return OutputParserParams(False, output_end_delimiter_ids=output_end_ids or [])

    # Enabled types: "forced" (the prompt ends inside the reasoning channel,
    # thinking cannot be turned off) and "triggered" (the model emits the
    # opening marker itself).
    if not reasoning_end_ids:
        raise ValueError("Invalid reasoning token id")
    if parser_type == "triggered" and not reasoning_start_ids:
        raise ValueError("Invalid reasoning token id")

    return OutputParserParams(
        True,
        reasoning_start_delimiter_ids=reasoning_start_ids or [],
        reasoning_end_delimiter_ids=reasoning_end_ids,
        output_end_delimiter_ids=output_end_ids or [],
        initial_state=(parser_type == "forced"),
    )


# use custom cache since in unit test we need to mock the result
_output_parser_params_cache: dict[bool, OutputParserParams] = {}


def get_output_parser_params(enable_thinking: bool) -> OutputParserParams:
    params = _output_parser_params_cache.get(enable_thinking)
    if params is None:
        params = _get_output_parser_params(enable_thinking)
        _output_parser_params_cache[enable_thinking] = params
    return params


def update_chat_template_kwargs_reasoning(kwargs: dict, enable_thinking: bool):
    """Normalize the reasoning flag(s) already present in `kwargs`.

    Which kwarg name a model's chat template reads (`thinking` vs
    `enable_thinking`) is decided solely by the serve layer's
    `build_chat_template_kwargs` (keyed by model name); this function only
    rewrites the *values* of whichever flags are present so they follow the
    effective reasoning state (e.g. a `forced` model keeps thinking on even
    for enable_thinking=False requests). It never adds or renames keys.
    """
    reasoning_params = get_output_parser_params(enable_thinking)
    KEYS = {"thinking", "enable_thinking"}
    for k in KEYS:
        if k in kwargs:
            kwargs[k] = reasoning_params.enable_reasoning
