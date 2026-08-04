from types import SimpleNamespace

from chitu.distributed import custom_ar_chitu


def test_all_ranks_true_uses_group_min(monkeypatch):
    seen = {}

    def fake_all_reduce(flag, *, op, group):
        seen.update(op=op, group=group)
        flag.zero_()

    monkeypatch.setattr(custom_ar_chitu.dist, "all_reduce", fake_all_reduce)
    group = object()
    assert custom_ar_chitu._all_ranks_true(group, True) is False
    assert seen == {"op": custom_ar_chitu.dist.ReduceOp.MIN, "group": group}


def test_varlen_collective_support_requires_hygon_and_both_symbols(monkeypatch):
    manager = object.__new__(custom_ar_chitu.ChituCustomAllreduce)
    manager.disabled = False
    manager._ptr = 0
    manager._is_hygon = True
    manager._supports_varlen_collectives = True
    monkeypatch.setattr(
        custom_ar_chitu,
        "chitu_backend",
        SimpleNamespace(
            hygon_varlen_collective_abi_version=lambda: 2,
            varlen_all_gather=lambda: None,
            varlen_reduce_scatter=lambda: None,
        ),
    )
    assert custom_ar_chitu._has_hygon_varlen_collective_api()
    assert manager.supports_varlen_collectives

    manager._is_hygon = False
    assert not manager.supports_varlen_collectives
    manager._is_hygon = True
    custom_ar_chitu.chitu_backend.varlen_reduce_scatter = None
    assert not custom_ar_chitu._has_hygon_varlen_collective_api()

    custom_ar_chitu.chitu_backend.varlen_reduce_scatter = lambda: None
    custom_ar_chitu.chitu_backend.hygon_varlen_collective_abi_version = lambda: 1
    assert not custom_ar_chitu._has_hygon_varlen_collective_api()
