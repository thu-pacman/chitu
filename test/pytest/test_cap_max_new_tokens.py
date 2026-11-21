# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Pin the output budget left after an instance is chosen.

The Router caps ``max_new_tokens`` against the global ``infer.max_seq_len``
before it knows which instance serves the request. Instances may be smaller, so
both call sites (``process_scheduler_request`` and the PD decode path) redo the
cap with the instance's own limit through this helper.
"""

import pytest

from chitu.utils import cap_max_new_tokens_for_instance


@pytest.mark.parametrize(
    "max_new_tokens,prefix_len,generated_len,max_seq_len,expected",
    [
        # The instance has room to spare: the requested budget survives.
        (1024, 100, 0, 8192, 1024),
        # Same request on a smaller instance: only 8200 - 8000 tokens are left.
        (2048, 8000, 0, 8200, 200),
        # `max_new_tokens` counts tokens already generated and `prefix_len`
        # already includes them, so the budget is not shrunk twice.
        (2048, 1000, 100, 8192, 1948),
        # Capacity and budget tie: keep the budget.
        (512, 512, 0, 1024, 512),
        # Exactly full: no room left.
        (2048, 4096, 0, 4096, 0),
        # Prompt longer than the instance: floor at zero. A negative budget
        # would corrupt the stop conditions downstream.
        (2048, 5000, 0, 4096, 0),
        # The whole budget was already spent before entering this instance.
        (2048, 3000, 2048, 8192, 0),
    ],
)
def test_cap_matches_the_instance_limit(
    max_new_tokens, prefix_len, generated_len, max_seq_len, expected
):
    assert (
        cap_max_new_tokens_for_instance(
            max_new_tokens, prefix_len, generated_len, max_seq_len
        )
        == expected
    )
