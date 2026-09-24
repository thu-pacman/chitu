# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Phone-book ("电话本") needle test helpers.

A phone-book prompt lists ``num_entries`` contacts (name + 11-digit number),
then asks for one specific entry's number. The prompt is far longer than a
single KV block — and, for sparse-attention models, than ``index_topk`` — so the
answer can only be produced if the right tokens were actually attended to (or
restored from the cache). A wrong answer therefore flags an indexer or
prefix-cache regression.

Two modes:

* **needle** — ``CHITU_TEST_PHONEBOOK=true``. Each request gets its own phone
  book (``seed=1234+i``), so no prompt in the batch is cacheable. This is the
  original DeepSeek indexer test (``deepseek_v3_2_indexer_needle_h20``).
* **shared prefix** — ``CHITU_TEST_PHONEBOOK_SHARED=true``. Every request shares
  the same body (one fixed seed) and only the final question differs, so the
  long prefix *is* cacheable. That is what makes this a prefix-cache correctness
  test: a request that hits the prefix cache must still answer with the right
  number, so a wrong page / wrong linear-attention checkpoint shows up as a
  failure instead of passing silently.

Both callers (``test/single_req_test.py`` for the mixed P+D case and
``chitu/testing/pd_utils.py`` for the disaggregated case) share this module, so
this file must stay importable without torch or any chitu runtime module.

Size the phone book with care: ``num_entries`` must keep the tokenized prompt
*plus the generated tokens* below the model's ``index_topk`` (2048 for
GLM-5.3-Flash) — see the long note in ``ci/platforms/h20/pd_test.yml``.
``IndexerGLM5Next._build_kpool_topk`` switches to a device→host-syncing
per-row loop once a decode batch is longer than ``index_topk``, which CUDA
graph capture cannot replay, so an over-long prompt here fails the run for a
reason that has nothing to do with prefix caching.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

_BASE_NAMES = [
    "张三",
    "李四",
    "王五",
    "赤兔",
    "八卦炉",
    "刘备",
    "关羽",
    "张飞",
    "赵云",
    "曹操",
    "曹丕",
    "曹植",
    "吕布",
    "貂蝉",
    "孙尚香",
    "孙权",
]

DEFAULT_NUM_ENTRIES = 500
#: Fixed seed used by the shared-prefix mode. Every request built with it has a
#: byte-identical body; only the question line varies.
SHARED_SEED = 1234


def gen_phonebook_body(num_entries: int, seed: int = SHARED_SEED):
    """Build the phone-book body and return ``(body, numbers)``.

    The body depends only on ``(num_entries, seed)`` — not on which entry is
    asked about — which is what lets several requests share a cacheable prefix.
    """
    rng = random.Random(seed)
    entries = []
    numbers = []
    used = set()
    for i in range(num_entries):
        name = f"{_BASE_NAMES[i % len(_BASE_NAMES)]}{i:04d}"
        # 11-digit phone number, unique per entry.
        while True:
            number = "1" + "".join(str(rng.randint(0, 9)) for _ in range(10))
            if number not in used:
                used.add(number)
                break
        numbers.append(number)
        entries.append(f"{name}\t{number}")
    return "\n".join(entries), numbers


def gen_phonebook_prompt(
    num_entries: int, exp_idx: int, seed: int = SHARED_SEED
) -> tuple[str, str]:
    """Build a phone-book prompt and return ``(prompt, expected_number)``.

    ``num_entries`` should be large enough that the tokenized prompt fills
    several full KV blocks (and exceeds ``index_topk``, 2048 for DeepSeek-V3.2),
    forcing the indexer's top-k to actually prune. ``exp_idx`` picks which entry
    is asked about.

    Output is byte-identical to the original ``test/single_req_test.py``
    implementation for the same arguments — ``test/pytest/test_phonebook.py``
    pins that down with a golden digest.
    """
    body, numbers = gen_phonebook_body(num_entries, seed)
    needle_name = f"{_BASE_NAMES[exp_idx % len(_BASE_NAMES)]}{exp_idx:04d}"
    expected_number = numbers[exp_idx]
    prompt = (
        "下面是一份电话簿，每行是一个联系人的姓名和电话号码，用制表符分隔。\n"
        "请仔细阅读，然后回答末尾的问题。\n\n"
        f"{body}\n\n"
        f"问题：{needle_name} 的电话号码是多少？请只输出这一串数字，不要输出其它内容。"
    )
    return prompt, expected_number


def phonebook_test_enabled() -> bool:
    """Whether to run the phone-book needle request instead of the usual msgs."""
    return os.environ.get("CHITU_TEST_PHONEBOOK", "false") == "true"


def phonebook_shared_enabled() -> bool:
    """Whether every request shares one phone-book body (prefix-cache mode)."""
    return os.environ.get("CHITU_TEST_PHONEBOOK_SHARED", "false") == "true"


def phonebook_mode_active() -> bool:
    """Whether any phone-book mode is on."""
    return phonebook_test_enabled() or phonebook_shared_enabled()


def phonebook_num_entries(default: int = DEFAULT_NUM_ENTRIES) -> int:
    """Number of phone-book entries, overridable via the environment.

    The prompt length scales linearly with this, so it is the knob to lower when
    the prompt does not fit in ``infer.max_seq_len``.
    """
    raw = os.environ.get("CHITU_TEST_PHONEBOOK_ENTRIES")
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            f"invalid CHITU_TEST_PHONEBOOK_ENTRIES={raw!r}, "
            f"falling back to {default}"
        )
        return default


def no_hit_tokens(req: Any) -> Optional[int]:
    """``req.num_hit_tokens`` (None when the engine never set it)."""
    return getattr(req, "num_hit_tokens", None)


def expected_pairs(reqs: Sequence[Any]) -> list[tuple[Any, str]]:
    """Adapter from requests to ``[(req, expected_number), ...]``.

    ``req._needle_expected`` is set by the request generators in
    ``test/single_req_test.py``. It is an undeclared attribute (``UserRequest``
    is a plain dataclass, so it is invisible to serialization and type checkers),
    which is why reading it is confined to this one place and every request
    created in the phone-book mode must carry it.
    """
    pairs = []
    for req in reqs:
        expected = getattr(req, "_needle_expected", None)
        if expected is None:
            raise AssertionError(
                f"request {getattr(req, 'request_id', '?')} has no "
                f"_needle_expected although a phone-book mode is active"
            )
        pairs.append((req, expected))
    return pairs


def check_phonebook_test_results(
    pairs: Sequence[tuple[Any, str]], *, context: str = "phonebook"
) -> None:
    """Assert every request's output contains its expected number.

    ``pairs`` is an explicit ``[(req, expected_number), ...]`` list. It is
    deliberately not derived from a duck-typed attribute here: a silently
    missing expectation would otherwise turn the whole check into a no-op.

    Raises ``AssertionError`` on a miss so the failure propagates to a nonzero
    exit code and fails CI.
    """
    misses = []
    for i, (req, expected) in enumerate(pairs):
        output = getattr(req, "output", None) or ""
        hit = expected in output
        logger.info(
            f"[{context}][answer] req[{i}] rid={getattr(req, 'request_id', '?')} "
            f"expected={expected} hit={hit} output={output!r}"
        )
        if not hit:
            misses.append((i, getattr(req, "request_id", "?"), expected, output))
    if misses:
        raise AssertionError(
            f"{context}: {len(misses)}/{len(pairs)} request(s) answered with the "
            f"wrong phone number; the indexer or the prefix-cache restore likely "
            f"picked the wrong tokens. misses={misses}"
        )


def check_prefix_cache_hit(
    pairs: Sequence[tuple[Any, Optional[str]]],
    *,
    required: bool,
    context: str = "phonebook",
) -> int:
    """Log per-request hit tokens and return the number of cache hits.

    When ``required`` is set, raise ``AssertionError`` unless at least one
    request has ``num_hit_tokens > 0`` — the caller enables it exactly when
    prefix caching is on *and* the prompts share a long cacheable prefix, so
    zero hits means the cache is not working (or the prompt cannot fill a
    block).
    """
    num_hit = 0
    for i, (req, expected) in enumerate(pairs):
        hit_tokens = no_hit_tokens(req)
        if hit_tokens is not None and hit_tokens > 0:
            num_hit += 1
        logger.info(
            f"[{context}][hit] req[{i}] rid={getattr(req, 'request_id', '?')} "
            f"prompt_len={getattr(req, 'prompt_len', '?')} "
            f"num_hit_tokens={hit_tokens} expected={expected}"
        )
    if required and num_hit == 0:
        raise AssertionError(
            f"{context}: no request had num_hit_tokens > 0 although prefix "
            f"caching is enabled and every prompt shares a long prefix "
            f"({len(pairs)} request(s) checked). Either the prefix cache is "
            f"broken or the prompt is shorter than one KV block."
        )
    logger.info(
        f"[{context}] prefix caching check "
        f"{'PASSED' if num_hit else 'skipped'}: "
        f"{num_hit}/{len(pairs)} request(s) hit the prefix cache"
    )
    return num_hit
