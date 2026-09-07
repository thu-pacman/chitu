# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from contextlib import suppress
from chitu.global_vars import get_global_args
from .type_def import ReasoningParams, ReasoningType

logger = logging.getLogger(__name__)


@functools.cache
def get_reasoning_token_id():
    from chitu.backend import Backend

    if Backend.tokenizer is None:
        logger.warning(
            "get reasoning token id before tokenizer initialized, only make sense in unit test"
        )
        return -1, -1

    with suppress(Exception):
        start_token_id, end_token_id = Backend.tokenizer.encode(
            "<think></think>", bos=False, eos=False
        )
        return start_token_id, end_token_id

    with suppress(Exception):
        start_token_id = int(Backend.args.models.get("rs_token_id", -1))
        end_token_id = int(Backend.args.models.get("re_token_id", -1))
        assert start_token_id != -1 and end_token_id != -1
        return start_token_id, end_token_id

    logger.info("This model does not support reasoning")
    return -1, -1


def _get_reasoning_params(enable_thinking: bool) -> ReasoningParams:
    try:
        args = get_global_args()
        reasoning_type: ReasoningType = args.models.reasoning_type
    except Exception:
        # Model not configured with a reasoning_type: fall back to "auto".
        logger.info('fallback to "auto" reasoning type')
        reasoning_type = "auto"

    if reasoning_type == "auto":
        start_token_id, end_token_id = get_reasoning_token_id()
        if start_token_id == -1 or end_token_id == -1:
            reasoning_type = "disabled"
        else:
            reasoning_type = "triggered"

    if reasoning_type == "switchable":
        reasoning_type = "forced" if enable_thinking else "disabled"

    if reasoning_type == "triggered":
        if not enable_thinking:
            reasoning_type = "disabled"

    if reasoning_type == "disabled":
        return ReasoningParams(False)

    if reasoning_type == "triggered":
        start_token_id, end_token_id = get_reasoning_token_id()
        if start_token_id == -1 or end_token_id == -1:
            raise ValueError("Invalid reasoning token id")
        return ReasoningParams(enable_thinking, start_token_id, end_token_id)

    if reasoning_type == "forced":
        start_token_id, end_token_id = get_reasoning_token_id()
        if end_token_id == -1:
            raise ValueError("Invalid reasoning token id")
        return ReasoningParams(True, start_token_id, end_token_id, initial_state=True)

    raise NotImplementedError(f"Unknown reasoning_type {repr(reasoning_type)}")


# use custom cache since in unit test we need to mock the result
_reasoning_params_cache: dict[bool, ReasoningParams] = {}


def get_reasoning_params(enable_thinking: bool) -> ReasoningParams:
    params = _reasoning_params_cache.get(enable_thinking)
    if params is None:
        params = _get_reasoning_params(enable_thinking)
        _reasoning_params_cache[enable_thinking] = params
    return params


def update_chat_template_kwargs_reasoning(kwargs: dict, enable_thinking: bool):
    reasoning_params = get_reasoning_params(enable_thinking)
    KEYS = {"thinking", "enable_thinking"}
    for k in KEYS:
        if k in kwargs:
            kwargs[k] = reasoning_params.enable_reasoning
