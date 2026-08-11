# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import contextlib
from xgrammar import (
    TokenizerInfo,
    GrammarCompiler,
    CompiledGrammar,
    Grammar,
)
import threading
import json

from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor

from chitu.global_vars import get_global_args
from chitu.tool_call.type_def import ToolCallParams
from chitu.tool_call import build_grammar

logger = logging.getLogger(__name__)


@functools.cache
def get_tokenizer_info():
    from chitu.backend import Backend

    try:
        tokenizer_info = TokenizerInfo.from_huggingface(
            Backend.tokenizer.model, vocab_size=Backend.args.models.vocab_size
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
_GRAMMAR_EXECUTOR = ThreadPoolExecutor(
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


def submit_grammar_compile(params: ToolCallParams) -> Future:
    """Return a Future resolving to the CompiledGrammar (or None) for this
    tool-schema.
        - completed hit -> a cached completed Future
        - in-flight hit -> the shared Future for a concurrent compile
        - cold miss     -> a new Future backed by the background compile pool
    Callers never block here; they read .result() only when the grammar is
    actually needed (sample time)
    """
    global _GRAMMAR_CACHE_LOOKUPS, _GRAMMAR_CACHE_HITS

    assert params is not None
    key = json.dumps(
        ToolCallParams.to_dict(params),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

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
