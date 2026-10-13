# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""The KV-transfer slowdown injection has to be gated by the test config.

It exists so the client-disconnect case can interrupt a request whose KV blocks
are demonstrably on the wire: the send of one request is milliseconds long on a
compressed-KV (MLA) model, far below what /dp/debug can be polled at.
"""

import time

from chitu.global_vars import set_global_args
from chitu.testing.exception import test_inject_exception_kv_transfer_slow
from omegaconf import OmegaConf


def _set_injection(name: str) -> None:
    set_global_args(
        OmegaConf.create({"test": {"inject_exception": name}}),
        need_ensure=False,
        need_preprocess=False,
    )


def test_kv_transfer_slow_injection_is_off_by_default(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    _set_injection("")

    test_inject_exception_kv_transfer_slow("req-test")

    assert slept == []


def test_kv_transfer_slow_injection_sleeps_when_configured(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    monkeypatch.setenv("CHITU_TEST_KV_TRANSFER_SLOW_S", "2.5")
    _set_injection("kv_transfer_slow")

    test_inject_exception_kv_transfer_slow("req-test")

    assert slept == [2.5]


def test_kv_transfer_slow_injection_ignores_other_injections(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    _set_injection("decode_side_failure")

    test_inject_exception_kv_transfer_slow("req-test")

    assert slept == []
