# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
import contextlib
import threading
import json
from collections import OrderedDict
from concurrent.futures import Future

from chitu.utils import DaemonThreadPoolExecutor, try_import_opt_dep
from chitu.global_vars import get_global_args
from chitu.tool_call.type_def import ToolCallParams, ResponseFormatParams
from chitu.tool_call import build_grammar

logger = logging.getLogger(__name__)

xgrammar, has_xgrammar = try_import_opt_dep("xgrammar", "grammar")
if has_xgrammar:
    from xgrammar import TokenizerInfo, GrammarCompiler, CompiledGrammar, Grammar
else:
    TokenizerInfo = GrammarCompiler = CompiledGrammar = Grammar = None


@functools.cache
def get_tokenizer_info():
    if not has_xgrammar:
        return None
    from chitu.backend import Backend

    try:
        # by default xgrammar only recognize HF tokenizer's `eos_token_id` as
        # the EOS token. But in reality there are also other tokens acting as
        # EOS tokens. So we need to set them explicitly to xgrammar via
        # `stop_token_ids`.
        stop_tokens = Backend.tokenizer.stop_tokens
        vocab_size = Backend.args.models.vocab_size
        assert isinstance(stop_tokens, set)
        assert len(stop_tokens) > 0
        assert all(0 <= t < vocab_size for t in stop_tokens)
        tokenizer_info = TokenizerInfo.from_huggingface(
            Backend.tokenizer.model,
            vocab_size=vocab_size,
            stop_token_ids=list(stop_tokens),
        )
        return tokenizer_info
    except Exception:
        logger.exception("Failed to get tokenizer info")
        return None


@functools.cache
def get_grammar_compiler():
    tokenizer_info = get_tokenizer_info()
    if tokenizer_info is None:
        return None
    return GrammarCompiler(get_tokenizer_info())


@functools.cache
def get_op_device():
    op_impl = None
    with contextlib.suppress(Exception):
        args = get_global_args()
        op_impl = args.infer.op_impl

    return "cpu" if op_impl == "cpu" else "cuda"


def compile_grammar(grammar: Grammar | None) -> CompiledGrammar | None:
    compiler = get_grammar_compiler()
    if compiler is None or grammar is None:
        return None

    try:
        compiled = compiler.compile_grammar(grammar)
    except Exception:
        logger.warning(
            "Failed to compile grammar via xgrammar, "
            "falling back to unconstrained generation",
            exc_info=True,
        )
        return None

    return compiled


# ---------------------------------------------------------------------------
# Grammar future cache (keyed on tool-schema) + deferred-compile pool.
#
# Each cached value is a Future. A pending Future represents an in-flight compile;
# a completed Future represents the cached CompiledGrammar result. The result may
# legitimately be None, so caching the Future avoids a separate miss sentinel.
# ---------------------------------------------------------------------------

_GRAMMAR_CACHE_LOCK = threading.Lock()
_GRAMMAR_FUTURES_MAX = 2048

_GRAMMAR_FUTURES: OrderedDict[str, Future] = OrderedDict()
_GRAMMAR_EXECUTOR = DaemonThreadPoolExecutor(
    max_workers=4, thread_name_prefix="grammar-compile"
)

_GRAMMAR_CACHE_LOOKUPS = 0
_GRAMMAR_CACHE_HITS = 0


def _done_future(result) -> Future:
    done = Future()
    done.set_result(result)
    return done


def _grammar_cache_hit_rate() -> float:
    if _GRAMMAR_CACHE_LOOKUPS == 0:
        return 0.0
    return _GRAMMAR_CACHE_HITS * 100.0 / _GRAMMAR_CACHE_LOOKUPS


def _compile_grammar_worker(params):
    """Runs on _GRAMMAR_EXECUTOR: grammar build and compile.

    The returned Future is stored in _GRAMMAR_FUTURES before this worker starts.
    Completed futures stay cached until LRU eviction.
    """
    grammar = build_grammar(params)
    return compile_grammar(grammar)


def _grammar_params_cache_key(params: ToolCallParams | ResponseFormatParams) -> str:
    """Stable cache key for either tools or response_format grammar parameters.

    The ``tools:`` / ``response_format:`` prefix keeps the two flavours' keys
    from colliding (both dump to plain dicts).
    """
    if isinstance(params, ToolCallParams):
        kind, payload = "tools", ToolCallParams.to_dict(params)
    elif isinstance(params, ResponseFormatParams):
        kind, payload = "response_format", ResponseFormatParams.to_dict(params)
    else:
        raise TypeError(f"Unsupported grammar params type: {type(params).__name__}")
    return (
        kind
        + ":"
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def submit_grammar_compile(
    params: ToolCallParams | ResponseFormatParams,
) -> Future:
    """Return a Future resolving to the CompiledGrammar (or None) for these
    grammar params (tool schemas or response_format).
        - completed hit -> a cached completed Future
        - in-flight hit -> the shared Future for a concurrent compile
        - cold miss     -> a new Future backed by the background compile pool
    Callers never block here; they read .result() only when the grammar is
    actually needed (sample time)
    """
    global _GRAMMAR_CACHE_LOOKUPS, _GRAMMAR_CACHE_HITS

    assert params is not None
    key = _grammar_params_cache_key(params)

    if get_grammar_compiler() is None:
        logger.debug(
            "[GRAMMAR] cache=skip reason=no_compiler hit_rate=%.1f%% cache=%d",
            _grammar_cache_hit_rate(),
            len(_GRAMMAR_FUTURES),
        )
        return _done_future(None)

    with _GRAMMAR_CACHE_LOCK:
        _GRAMMAR_CACHE_LOOKUPS += 1
        fut = _GRAMMAR_FUTURES.get(key)
        if fut is not None:
            _GRAMMAR_FUTURES.move_to_end(key)
            _GRAMMAR_CACHE_HITS += 1
            logger.debug(
                "[GRAMMAR] cache=hit hit_rate=%.1f%% cache=%d",
                _grammar_cache_hit_rate(),
                len(_GRAMMAR_FUTURES),
            )
            return fut

        fut = _GRAMMAR_EXECUTOR.submit(_compile_grammar_worker, params)
        _GRAMMAR_FUTURES[key] = fut
        while len(_GRAMMAR_FUTURES) > _GRAMMAR_FUTURES_MAX:
            _GRAMMAR_FUTURES.popitem(last=False)
        logger.debug(
            "[GRAMMAR] cache=miss hit_rate=%.1f%% cache=%d",
            _grammar_cache_hit_rate(),
            len(_GRAMMAR_FUTURES),
        )
        return fut


def validate_response_format_grammar(params: ResponseFormatParams) -> str | None:
    """Check that a response_format grammar can be enforced on this server.

    This is an early check to return HTTP 4xx in time. The actual grammar
    enforcement is deferred to sample time.
    """
    if get_grammar_compiler() is None:
        return (
            "response_format is not available on this server: "
            "the grammar compiler could not be initialized"
        )
    if submit_grammar_compile(params).result() is None:
        return (
            "response_format could not be compiled into an enforceable "
            "grammar; please check that the schema is a supported JSON Schema"
        )
    return None
