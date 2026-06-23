import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import chitu.chitu_main as chitu_main
from chitu.ops.utils import (
    emit_observed_op_impl_summary,
    format_observed_op_impl_summary_lines,
    make_op_dispatcher,
    reset_observed_op_impl_state,
)


@pytest.fixture(autouse=True)
def reset_observed_state():
    reset_observed_op_impl_state()
    yield
    reset_observed_op_impl_state()


def test_make_op_dispatcher_tracks_selected_and_available_impls():
    @make_op_dispatcher
    def fused_op(x, *, impl="auto"):
        raise NotImplementedError

    @fused_op.register_auto
    def _auto_fused_op():
        return "triton"

    @fused_op.register("torch")
    def _fused_op_torch(x):
        return ("torch", x)

    @fused_op.register("triton")
    def _fused_op_triton(x):
        return ("triton", x)

    @fused_op.register("cuda")
    def _fused_op_cuda(x):
        return ("cuda", x)

    @fused_op.register("torch_npu", available=False)
    def _fused_op_torch_npu(x):
        return ("torch_npu", x)

    assert fused_op(3) == ("triton", 3)
    assert fused_op(4, impl="torch") == ("torch", 4)

    with pytest.raises(NotImplementedError, match="not available in current env"):
        fused_op(5, impl="torch_npu")

    assert format_observed_op_impl_summary_lines(pretty=False) == [
        "fused_op | cuda · | torch ✓ | torch_npu × | triton ✓"
    ]


def test_emit_observed_op_impl_summary_logs_once(caplog):
    @make_op_dispatcher
    def op(x, *, impl="auto"):
        raise NotImplementedError

    @op.register_auto
    def _auto_op():
        return "torch"

    @op.register("torch")
    def _op_torch(x):
        return x

    op(1)

    test_logger = logging.getLogger("test.op_impl_dispatcher")
    with caplog.at_level(logging.INFO, logger=test_logger.name):
        assert emit_observed_op_impl_summary(target_logger=test_logger) is True
        assert emit_observed_op_impl_summary(target_logger=test_logger) is False

    caplog.messages[:-1] == [
        "Dispatched `op` to `torch`. (set CHITU_LOG_STACK_TRACE=1 for call site)",
        "Operator implementations used during warmup (✓ = used at least once; · = not used; × = not installed):"
        "op | torch ✓",
    ]


def test_make_op_dispatcher_supports_custom_op_name():
    @make_op_dispatcher(op_name="public_op")
    def internal_impl(x, *, impl="auto"):
        raise NotImplementedError

    @internal_impl.register_auto
    def _auto_internal_impl():
        return "torch"

    @internal_impl.register("torch")
    def _internal_impl_torch(x):
        return x

    internal_impl(1)

    assert format_observed_op_impl_summary_lines(pretty=False) == [
        "public_op | torch ✓"
    ]


def test_auto_resolver_filters_impl_kwarg():
    @make_op_dispatcher
    def op(x, *, scale=1, impl="auto"):
        raise NotImplementedError

    @op.register_auto
    def _auto_op(x, *, scale=1):
        assert scale == 2
        return "torch"

    @op.register("torch")
    def _op_torch(x, *, scale=1):
        return x * scale

    assert op(3, scale=2) == 6
    assert format_observed_op_impl_summary_lines(pretty=False) == ["op | torch ✓"]


def test_auto_resolver_keyword_only_params_not_passed_positional_args():
    @make_op_dispatcher
    def op(a, b, *, flag=False, impl="auto"):
        raise NotImplementedError

    @op.register_auto
    def _auto_op(*, flag=False):
        return "fast" if flag else "slow"

    @op.register("fast")
    def _op_fast(a, b, *, flag=False):
        return a + b

    @op.register("slow")
    def _op_slow(a, b, *, flag=False):
        return a - b

    assert op(10, 3, flag=True) == 13
    assert op(10, 3, flag=False) == 7
    assert op(10, 3) == 7


def test_warmup_engine_emits_impl_summary_after_auto_set(monkeypatch):
    events = []

    monkeypatch.setattr(
        chitu_main, "clear_observed_op_impl_selections", lambda: events.append("clear")
    )
    monkeypatch.setattr(
        chitu_main, "_warmup_via_taskpool", lambda args: events.append("warmup")
    )
    monkeypatch.setattr(
        chitu_main,
        "_auto_set_num_blocks_after_warmup",
        lambda args: events.append("auto_set"),
    )
    monkeypatch.setattr(
        chitu_main,
        "_emit_observed_op_impl_summary_after_warmup",
        lambda: events.append("emit"),
    )

    monkeypatch.setattr(chitu_main, "is_classic_pd_disagg", lambda: False)
    monkeypatch.setattr(chitu_main, "is_independent_multi_inst", lambda: True)

    args = SimpleNamespace(
        multi_inst=SimpleNamespace(
            n_insts=2,
            router=SimpleNamespace(is_router=False),
        ),
        scheduler=SimpleNamespace(type="default"),
        infer=SimpleNamespace(full_warmup=False, prefill_chunk_size=100),
    )

    chitu_main.warmup_engine(args)

    assert events == ["clear", "warmup", "auto_set", "emit"]


def test_emit_observed_op_impl_summary_after_warmup_skips_nonzero_rank(monkeypatch):
    monkeypatch.setattr(chitu_main.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(chitu_main.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(chitu_main.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(
        chitu_main,
        "emit_observed_op_impl_summary",
        lambda target_logger=None: pytest.fail("summary should not be emitted"),
    )

    assert chitu_main._emit_observed_op_impl_summary_after_warmup() is False


def test_ops_source_tree_has_no_handwritten_auto_impl_dispatch_left():
    ops_root = Path(__file__).resolve().parents[2] / "chitu" / "ops"
    offenders = []

    for path in ops_root.rglob("*.py"):
        if path.name == "utils.py" or "triton_ops" in path.parts:
            continue
        if 'if impl == "auto":' in path.read_text():
            offenders.append(path.relative_to(ops_root.parent).as_posix())

    assert offenders == []


def test_blockfp_quantization_files_have_no_handwritten_auto_impl_dispatch_left():
    quant_root = Path(__file__).resolve().parents[2] / "chitu" / "quantization"
    offenders = []

    for relative_path in ("blockfp4.py", "blockfp8.py"):
        path = quant_root / relative_path
        if not path.exists():
            continue
        if 'impl == "auto"' in path.read_text():
            offenders.append(relative_path)

    assert offenders == []


def test_tail_cleanup_files_have_no_handwritten_impl_branching_left():
    repo_root = Path(__file__).resolve().parents[2] / "chitu"
    offenders = []

    for relative_path in (
        "quantization/normal.py",
        "ops/sampling.py",
        "ops/rotary.py",
    ):
        path = repo_root / relative_path
        if not path.exists():
            continue
        text = path.read_text()
        if "if impl ==" in text or "elif impl ==" in text:
            offenders.append(relative_path)

    assert offenders == []
