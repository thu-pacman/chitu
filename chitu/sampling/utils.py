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

from chitu.global_vars import get_global_args

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


def compile_grammar(grammar: Grammar | None) -> tuple[CompiledGrammar | None, str]:
    compiler = get_grammar_compiler()
    if compiler is None or grammar is None:
        return None, ""

    compiled = compiler.compile_grammar(grammar)
    grammar_str = compiled.serialize_json()
    return compiled, grammar_str


def deserialize_grammar(grammar_str: str) -> CompiledGrammar | None:
    if grammar_str:
        return CompiledGrammar.deserialize_json(grammar_str, get_tokenizer_info())
    return None
