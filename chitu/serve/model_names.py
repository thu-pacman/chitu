# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Accepted names for single-model serving; never select model weights."""


def available_model_names(args) -> list[str]:
    loaded = args.models.name
    alias = getattr(args.serve, "model_alias", None)
    return [loaded, alias] if alias and alias != loaded else [loaded]


def resolve_model_name(requested_model: str | None, args) -> str:
    if requested_model is None:
        return args.models.name
    if requested_model in available_model_names(args):
        return requested_model
    raise ValueError(f"Model '{requested_model}' is not available on this server.")
