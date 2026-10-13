import torch
from torch import nn

import chitu.models.model_hf_qwen3_next as qwen3_next


class _FakeBlockfp8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, tp_size: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.local_out_features = out_features
        self.block_size = 128
        self.round_scale_to_pow2 = False
        self.tp_size = tp_size
        self.gather_output = False
        values = torch.arange(out_features * in_features, dtype=torch.float32)
        values = ((values % 31) - 15).reshape(out_features, in_features) / 16
        self.weight = nn.Parameter(
            values.to(torch.float8_e4m3fn), requires_grad=False
        )
        self.scale = nn.Parameter(
            torch.ones(out_features // 128, in_features // 128),
            requires_grad=False,
        )
        self.register_parameter("bias", None)

    def forward(self, x):
        return x.float() @ self.weight.float().T


def _attention(qkv, gate):
    attention = nn.Module()
    attention.qkv_proj = qkv
    attention.attn_gate = gate
    attention._gcu_qkv_gate_merge_state = None
    return attention


def test_qkv_gate_merge_is_exact_and_preserves_public_state_dict(monkeypatch):
    monkeypatch.setattr(qwen3_next, "Blockfp8Linear", _FakeBlockfp8Linear)
    qkv = _FakeBlockfp8Linear(128, 256)
    gate = _FakeBlockfp8Linear(128, 128)
    attention = _attention(qkv, gate)
    x = torch.arange(256, dtype=torch.float32).reshape(2, 128) / 256
    expected_qkv = qkv(x)
    expected_gate = gate(x)
    expected_keys = set(attention.state_dict())

    merged, qkv_n = (
        qwen3_next.AttentionQwen3Next._get_gcu_qkv_gate_merged_projection(
            attention
        )
    )
    actual = merged(x)

    torch.testing.assert_close(actual[:, :qkv_n], expected_qkv, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, qkv_n:], expected_gate, rtol=0, atol=0)
    assert set(attention.state_dict()) == expected_keys
    assert tuple(attention.qkv_proj.weight.shape) == (256, 128)
    assert tuple(attention.attn_gate.weight.shape) == (128, 128)
    assert attention.qkv_proj.weight.data_ptr() == merged.weight.data_ptr()
    assert attention.attn_gate.weight.data_ptr() == (
        merged.weight.data_ptr() + qkv_n * qkv.in_features
    )


def test_qkv_gate_merge_fails_closed_for_tp2(monkeypatch):
    monkeypatch.setattr(qwen3_next, "Blockfp8Linear", _FakeBlockfp8Linear)
    attention = _attention(
        _FakeBlockfp8Linear(128, 256, tp_size=2),
        _FakeBlockfp8Linear(128, 128, tp_size=2),
    )

    assert (
        qwen3_next.AttentionQwen3Next._get_gcu_qkv_gate_merged_projection(
            attention
        )
        is None
    )
    assert attention._gcu_qkv_gate_merge_state is False
