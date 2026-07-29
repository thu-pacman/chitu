# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""MiniMax M3 sparse attention path selection (resolved once at init)."""

from __future__ import annotations

from logging import getLogger
from typing import Literal, Optional

logger = getLogger(__name__)

MiniMaxSparsePrefillBackend = Literal["ref", "dense_flash", "triton"]
MiniMaxSparsePrefillBackendConfig = Literal["dense_flash", "triton", "auto"]
MiniMaxSparseDecodeBackend = Literal["remap", "triton"]

# Best crossover threshold measured on H20 (Triton vs dense Flash prefill).
SPARSE_PREFILL_AUTO_THRESHOLD = 9216

_RESOLVED_PREFILL_BACKEND_CONFIG: Optional[
    MiniMaxSparsePrefillBackendConfig | Literal["ref"]
] = None
_RESOLVED_DECODE_BACKEND: Optional[MiniMaxSparseDecodeBackend] = None


def _max_prefill_query_len(seq_len_delta) -> int:
    max_q = 0
    for i in range(seq_len_delta.batch_size):
        q_len = (
            seq_len_delta.delta_prefix_lens_list[i + 1]
            - seq_len_delta.delta_prefix_lens_list[i]
        )
        max_q = max(max_q, q_len)
    return max_q


def _resolve_decode_backend() -> MiniMaxSparseDecodeBackend:
    from chitu.global_vars import get_global_args

    backend = str(
        getattr(get_global_args().infer, "minimax_sparse_decode_backend", "remap")
    )
    if backend not in ("remap", "triton"):
        raise ValueError(
            f"Unsupported minimax_sparse_decode_backend={backend!r}; "
            "expected 'remap' or 'triton'"
        )
    return backend  # type: ignore[return-value]


def _resolve_prefill_backend_config(
    attn_backend,
) -> MiniMaxSparsePrefillBackendConfig | Literal["ref"]:
    from chitu.attn_backend.ref_attn_backend import RefAttnBackend

    if isinstance(attn_backend, RefAttnBackend):
        return "ref"

    from chitu.global_vars import get_global_args

    backend = str(
        getattr(get_global_args().infer, "minimax_sparse_prefill_backend", "auto")
    )
    if backend not in ("dense_flash", "triton", "auto"):
        raise ValueError(
            f"Unsupported minimax_sparse_prefill_backend={backend!r}; "
            "expected 'dense_flash', 'triton', or 'auto'"
        )
    return backend  # type: ignore[return-value]


def ensure_minimax_sparse_attention_paths_initialized(attn_backend) -> None:
    """Resolve sparse prefill/decode backends once before layer construction."""
    global _RESOLVED_PREFILL_BACKEND_CONFIG
    global _RESOLVED_DECODE_BACKEND
    if _RESOLVED_PREFILL_BACKEND_CONFIG is not None:
        return

    _RESOLVED_PREFILL_BACKEND_CONFIG = _resolve_prefill_backend_config(attn_backend)
    if _RESOLVED_PREFILL_BACKEND_CONFIG == "ref":
        logger.info_once("MiniMax M3 sparse prefill: ref block mask")
    elif _RESOLVED_PREFILL_BACKEND_CONFIG == "dense_flash":
        logger.info_once("MiniMax M3 sparse prefill: dense Flash attention")
    elif _RESOLVED_PREFILL_BACKEND_CONFIG == "triton":
        logger.info_once("MiniMax M3 sparse prefill: Triton block sparse")
    else:
        logger.info_once(
            "MiniMax M3 sparse prefill: auto "
            f"(triton when max_query_len>={SPARSE_PREFILL_AUTO_THRESHOLD}, else dense Flash)"
        )

    _RESOLVED_DECODE_BACKEND = _resolve_decode_backend()
    if _RESOLVED_DECODE_BACKEND == "remap":
        logger.info_once(
            "MiniMax M3 sparse decode: block remap + configured attention backend"
        )
    else:
        logger.info_once("MiniMax M3 sparse decode: Triton per-KV-head block sparse")


def get_minimax_sparse_prefill_backend_config() -> (
    MiniMaxSparsePrefillBackendConfig | Literal["ref"]
):
    if _RESOLVED_PREFILL_BACKEND_CONFIG is None:
        raise RuntimeError(
            "MiniMax sparse prefill backend is unset; call "
            "ensure_minimax_sparse_attention_paths_initialized during model init"
        )
    return _RESOLVED_PREFILL_BACKEND_CONFIG


def resolve_minimax_sparse_prefill_backend(
    seq_len_delta,
) -> MiniMaxSparsePrefillBackend:
    """Pick the sparse prefill backend for the current step."""
    config = get_minimax_sparse_prefill_backend_config()
    if config == "ref":
        return "ref"
    if config == "triton":
        return "triton"
    if config == "dense_flash":
        return "dense_flash"

    max_q = _max_prefill_query_len(seq_len_delta)
    return "triton" if max_q >= SPARSE_PREFILL_AUTO_THRESHOLD else "dense_flash"


def get_minimax_sparse_prefill_mode() -> MiniMaxSparsePrefillBackend:
    """Return configured prefill mode (legacy name; does not resolve ``auto``)."""
    config = get_minimax_sparse_prefill_backend_config()
    if config == "auto":
        return "dense_flash"
    return config  # type: ignore[return-value]


def get_minimax_sparse_decode_backend() -> MiniMaxSparseDecodeBackend:
    if _RESOLVED_DECODE_BACKEND is None:
        raise RuntimeError(
            "MiniMax sparse decode backend is unset; call "
            "ensure_minimax_sparse_attention_paths_initialized during model init"
        )
    return _RESOLVED_DECODE_BACKEND
