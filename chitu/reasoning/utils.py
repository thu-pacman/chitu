# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from chitu.global_vars import get_global_args
from .type_def import ReasoningParams, ReasoningType

logger = logging.getLogger(__name__)


@functools.cache
def get_reasoning_token_id():
    from chitu.backend import Backend

    try:
        start_token_id, end_token_id = Backend.tokenizer.encode(
            "<think></think>", bos=False, eos=False
        )
        return start_token_id, end_token_id
    except:
        logger.exception("failed to get token id by tokenizer")

    try:
        start_token_id = int(Backend.args.models.get("rs_token_id", -1))
        end_token_id = int(Backend.args.models.get("re_token_id", -1))
        return start_token_id, end_token_id
    except:
        logger.exception("failed to get token id by config")

    return -1, -1


@functools.cache
def get_reasoning_params(enable_thinking: bool) -> ReasoningParams:
    args = get_global_args()
    try:
        reasoning_type: ReasoningType = args.models.reasoning_type
    except:
        logger.warning('Failed to get reasoning_type, fallback to "auto"')
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


def update_chat_template_kwargs_reasoning(
    kwargs: dict, reasoning_params: ReasoningParams
):
    for key in ["thinking", "enable_thinking"]:
        if key in kwargs:
            kwargs[key] = reasoning_params.enable_reasoning
