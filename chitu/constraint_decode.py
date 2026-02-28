# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from xgrammar import (
    apply_token_bitmask_inplace,
    allocate_token_bitmask,
    Grammar,
    CompiledGrammar,
    GrammarMatcher,
    TokenizerInfo,
    GrammarCompiler,
    BatchGrammarMatcher,
)
from chitu.device_type import is_ascend, is_muxi
import torch
import logging
from dataclasses import dataclass
from abc import ABC

logger = logging.getLogger()


@dataclass
class MatcherState:
    matcher: GrammarMatcher
    accepted_len: int = 0


class ConstraintDecodeTask(ABC):
    task_id: str
    prefix_tokens: list[int]
    prompt_len: int
    grammar_str: str
    grammar: CompiledGrammar | None


class ConstraintDecodeManager:
    def __init__(self, tokenizer, vocab_size: int):
        self.enabled = False
        try:
            self.tokenizer_info = TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=vocab_size
            )
            self.grammar_compiler = GrammarCompiler(self.tokenizer_info)
            self.batch_matcher = BatchGrammarMatcher()
            self.states: dict[str, MatcherState] = {}
            self.enabled = True
        except Exception:
            logger.exception("constraint decode initialized failed")

    def compile_grammar(
        self, grammar: Grammar | None
    ) -> tuple[CompiledGrammar | None, str]:
        if grammar is None or not self.enabled:
            return None, ""
        compiled_grammar = self.grammar_compiler.compile_grammar(grammar)
        grammar_str = compiled_grammar.serialize_json()
        return compiled_grammar, grammar_str

    def deserialize_grammar(self, grammar_str: str) -> CompiledGrammar | None:
        if grammar_str:
            return CompiledGrammar.deserialize_json(grammar_str, self.tokenizer_info)
        return None

    def apply_grammars(self, logits: torch.Tensor, tasks: list[ConstraintDecodeTask]):
        if not self.enabled:
            return
        matchers, indices = self._get_matchers(tasks)
        if len(matchers) == 0:
            return
        bitmask = allocate_token_bitmask(
            logits.shape[0], self.tokenizer_info.vocab_size
        )
        self.batch_matcher.batch_fill_next_token_bitmask(matchers, bitmask, indices)
        apply_bitmask(logits, bitmask.to(logits.device, non_blocking=True), indices)

    def _get_matchers(
        self, tasks: list[ConstraintDecodeTask]
    ) -> tuple[list[GrammarMatcher], list[int]]:
        matchers = []
        indices = []
        for i, task in enumerate(tasks):
            if task.grammar is None:
                continue
            state = self._get_state(task)
            for token in task.prefix_tokens[state.accepted_len :]:
                if state.matcher.is_terminated():
                    break
                state.matcher.accept_token(token)
            if not state.matcher.is_terminated():
                matchers.append(state.matcher)
                indices.append(i)
            state.accepted_len = len(task.prefix_tokens)
        return matchers, indices

    def _get_state(self, task: ConstraintDecodeTask):
        if task.task_id not in self.states:
            matcher = GrammarMatcher(task.grammar)
            self.states[task.task_id] = MatcherState(matcher, task.prompt_len)
        return self.states[task.task_id]

    def end_tasks(self, task_ids: list[str]):
        if not self.enabled:
            return
        for task_id in task_ids:
            self.states.pop(task_id, None)


def apply_bitmask_torch(
    logits: torch.Tensor, bitmask: torch.Tensor, indices: list[int]
):
    _, H = logits.shape
    _, M = bitmask.shape
    B = len(indices)
    bitmask = bitmask[indices].view(B, M, 1)
    # shift left fallback to cpu on npu, thus we use pow
    bits = torch.arange(32, device=logits.device, dtype=torch.int32)
    bits = torch.pow(2, bits).view(1, 1, 32)
    mask = bits & bitmask
    mask = mask.view(B, M * 32)[:, :H]
    logits[indices] = logits[indices].masked_fill_(mask == 0, float("-inf"))
    return logits


def apply_bitmask(logits: torch.Tensor, bitmask: torch.Tensor, indices: list[int]):
    if is_ascend() or is_muxi():
        return apply_bitmask_torch(logits, bitmask, indices)
    return apply_token_bitmask_inplace(logits, bitmask, indices=indices)
