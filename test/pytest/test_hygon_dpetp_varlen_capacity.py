# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from chitu.moe.impl import _compute_dp_etp_global_capacity


@pytest.mark.parametrize(
    (
        "max_batch_size",
        "dp_size",
        "pp_size",
        "decode_limit",
        "cache_type",
        "expected",
    ),
    [
        (96, 8, 4, "max", None, 24),
        (128, 8, 4, "max", None, 32),
        (144, 8, 4, "max", None, 40),
        (160, 8, 4, "max", None, 40),
        (160, 8, 1, "max", None, 160),
        (160, 8, 4, "3", None, 24),
        (160, 8, 4, "3", "skew", 40),
        (160, 8, 4, 30, None, 160),
        (10, 3, 2, "max", None, 6),
    ],
)
def test_compute_dp_etp_global_capacity(
    max_batch_size, dp_size, pp_size, decode_limit, cache_type, expected
):
    assert (
        _compute_dp_etp_global_capacity(
            max_batch_size,
            dp_size,
            pp_size,
            decode_limit,
            cache_type,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("max_batch_size", "dp_size", "pp_size", "decode_limit"),
    [
        (0, 8, 4, "max"),
        (96, 0, 4, "max"),
        (96, 8, 0, "max"),
        (96, 8, 4, "invalid"),
        (96, 8, 4, "0"),
    ],
)
def test_compute_dp_etp_global_capacity_rejects_invalid_values(
    max_batch_size, dp_size, pp_size, decode_limit
):
    with pytest.raises(ValueError):
        _compute_dp_etp_global_capacity(
            max_batch_size,
            dp_size,
            pp_size,
            decode_limit,
        )
