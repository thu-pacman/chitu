# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ``build_chat_template_kwargs`` reasoning_effort support.

``build_chat_template_kwargs`` is the common helper all serve paths (OpenAI /
Responses / ``/tokenize``) funnel through to build the kwargs passed to the chat
template.  These tests cover the ``reasoning_effort`` argument added on the
``reasoning-effort`` branch:

- backward compatibility (omitting the argument reproduces the old output),
- ``reasoning_effort=None`` keeping the key absent (so templates that guard on
  ``reasoning_effort is defined`` behave exactly as before),
- explicit values being injected verbatim (the serve layer does not validate).

The model name is patched to ``DeepSeek-V4-Flash`` — the model this branch is
validated against (GLM-5.2's config is not on this branch).  The value only has
to avoid the ``DeepSeek-V3.1`` special-case in the helper; the reasoning_effort
behaviour is identical for every non-V3.1 model.
"""

from types import SimpleNamespace

import pytest

import chitu.serve.common as serve_common


def _patch_model_name(monkeypatch, name):
    monkeypatch.setattr(
        serve_common,
        "get_global_args",
        lambda: SimpleNamespace(models=SimpleNamespace(name=name)),
    )


class TestBuildChatTemplateKwargs:
    # --- backward compatibility ---
    def test_omitting_reasoning_effort_matches_legacy_output(self, monkeypatch):
        _patch_model_name(monkeypatch, "DeepSeek-V4-Flash")
        assert serve_common.build_chat_template_kwargs(True) == {
            "enable_thinking": True
        }
        assert serve_common.build_chat_template_kwargs(False) == {
            "enable_thinking": False
        }

    def test_none_keeps_key_absent(self, monkeypatch):
        """``reasoning_effort=None`` must produce identical output to omission."""
        _patch_model_name(monkeypatch, "DeepSeek-V4-Flash")
        kwargs = serve_common.build_chat_template_kwargs(True, None)
        assert "reasoning_effort" not in kwargs
        assert kwargs == serve_common.build_chat_template_kwargs(True)

    # --- explicit values ---
    @pytest.mark.parametrize("effort", ["high", "max", "low", "medium", "weird"])
    def test_explicit_value_is_injected(self, monkeypatch, effort):
        _patch_model_name(monkeypatch, "DeepSeek-V4-Flash")
        kwargs = serve_common.build_chat_template_kwargs(True, effort)
        assert kwargs == {"enable_thinking": True, "reasoning_effort": effort}

    def test_reasoning_effort_independent_of_enable_thinking(self, monkeypatch):
        _patch_model_name(monkeypatch, "DeepSeek-V4-Flash")
        kwargs = serve_common.build_chat_template_kwargs(False, "high")
        assert kwargs == {"enable_thinking": False, "reasoning_effort": "high"}
