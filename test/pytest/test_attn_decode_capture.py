# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""The decode-prepare-in-a-capture contract, one case per backend and variant.

A decode step is one `reserve_metadata_for_decode` -- outside any capture --
followed by one `prepare_metadata_for_decode` + attention pair per phase: one
pair for a plain decode step, or a verify phase carrying `mtp_size` query tokens
per sequence plus the `mtp_size - 1` one-token draft phases of an MTP step.
`AttnBackend.decode_supports_prepare_in_graph` answers whether every one of those
prepares may run inside one captured region, which is what lets a whole MTP
iteration share one replay (`infer.mtp_draft_single_graph`).

This file is the executable mirror of `chitu/attn_backend/README.md`'s support
matrix: one case per backend and per configuration that changes its answer,
including the answers that only exist on another platform. A case is a
backend, a KV-cache layout (MLA or grouped-query), a page size and the two
flags the matrix documents for it. Cases this machine cannot build -- a package
it does not have, a platform-only configuration such as Hygon's e5m2 cache --
are skipped with the reason named, so the file states the whole matrix wherever
it runs and verifies as much of it as the machine can construct.

Three layers, from cheapest to strongest:

- `test_reserve_metadata_for_prefill_is_the_base_no_op_for_every_backend` and
  `test_only_the_backends_with_something_to_do_replace_a_decode_hook` read the
  classes themselves: none of them overrides the prefill reservation, and only
  the documented ones replace a decode hook at all. A backend that grew a
  host-touching decode prepare would change these sets, so it cannot slip in
  behind a flag that still says True.
- `test_decode_prepare_flags_match_the_documented_answer` builds every case and
  checks both flags -- the in-graph answer and whether the decode op takes a
  verify phase -- against what the matrix documents for this platform.
- `test_one_reserve_then_every_phase_in_one_capture` is the promise itself, for
  every case that claims it: reserve the verify and the draft shape outside the
  capture, warm the plans up outside too, then capture a region holding the
  verify phase and both draft steps -- each with its own
  `prepare_metadata_for_decode`, the draft loop advancing the delta's device
  lengths by one per phase -- and replay it at two quite different positions,
  comparing phase by phase with the eager result for the lengths that phase
  runs at. The two positions are far apart in KV span and page count, and the
  test first checks that the phases really differ between them, so a prepare
  that had baked the recording's lengths cannot pass.

The two `flash_infer_mla` tests cover the other half of the same contract for
the one backend whose reservation does something:
`test_flashinfer_mla_plans_a_reserved_shape_only_outside_a_capture` checks that
reserving is what makes the later prepares legal, and
`test_flashinfer_mla_reserved_drafts_match_the_host_planned_ones` checks that
the wider reserved plan scores what the host-built plan scores.
"""

import contextlib
from dataclasses import dataclass
from typing import Callable, Optional
import warnings

from omegaconf import OmegaConf
import packaging.version
import pytest
import torch

from chitu import cuda_graph as chitu_cuda_graph
from chitu.attn_backend import (
    AttnBackend,
    FlashAttnBackend,
    FlashInferBackend,
    FlashMLABackend,
    HopperMixedBackend,
    HunyuanAttnBackend,
    HybridAttnBackend,
    NpuAttnBackend,
    RefAttnBackend,
    TritonAttnBackend,
)
from chitu.attn_backend.dllm_backend import DLLMAttnBackend
from chitu.attn_backend.flash_mla_backend import (
    has_flash_mla_e5m2,
    has_flash_mla_sched_meta,
)
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.device_type import has_accelerator, is_hygon, is_muxi
from chitu.global_vars import set_global_args
from chitu.kv_cache import PagedKVCacheAccessor
from chitu.testing import assert_close
from chitu.utils import (
    try_import_and_setup_torch_npu,
    try_import_opt_dep,
    try_import_platform_dep,
)

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")
flash_attn3, has_flash_attn3 = try_import_opt_dep(
    "flash_attn_interface", "flash_attn_interface"
)
flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")
flash_mla, has_flash_mla = try_import_opt_dep("flash_mla", "flash_mla")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
hunyuan_ops, has_hunyuan_ops = try_import_opt_dep("hpc", "hpc_ops")

# `chitu.ops.triton_ops` imports triton at module level, so it cannot be
# imported on a platform without it -- Ascend, for one. Only the fp8 MLA cache
# case needs it, and that case is skipped there.
triton, has_triton = try_import_platform_dep("triton")
has_triton_ops = has_triton and has_accelerator()
if has_triton_ops:
    from chitu.ops.triton_ops import quant_pertoken_kvcache_dsa

needs_accelerator = pytest.mark.skipif(
    not has_accelerator(), reason="requires an accelerator"
)

BS = 2
MTP_SIZE = 3
# The KV range a page table covers, which is also the span a reservation below
# has to be wide enough for (flashinfer's plan asserts on it).
KV_SPAN = 1024
N_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 128
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
KV_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
# MLA's dense decode path wants 64-byte pages. FlashAttention's paged KV wants
# a multiple of 256 on CUDA, while the Hygon port wants 64 -- `test_attn.py`
# builds 64 there for the same kernel.
BLOCK_MLA = 64
BLOCK_GQA = 64 if is_hygon() else 256
# The sparse MLA decode paths take this many TopK positions per query row;
# production feeds `models.index_topk`.
TOPK = 2048
# Where a run starts: the verify phase attends up to this position. The two are
# far apart in KV span and page count, so a phase scored with some other
# phase's length cannot come out right, and a shorter one cannot pass for a
# longer one.
EARLY = 320
LATE = 640


def _rand(generator, *shape):
    return torch.randn(*shape, device="cuda", generator=generator)


def _flash_mla_in_graph() -> bool:
    """FlashMLA's documented answer, from this box's package capability.

    Nvidia has always shipped the device-only `FlashMLASchedMeta` interface,
    Hygon ships it in recent packages (a package capability, not a platform),
    and Muxi has not been checked either way.
    """
    if is_hygon():
        return has_flash_mla_sched_meta
    return not is_muxi()


def _flash_attn_available() -> bool:
    return has_flash_attn or has_flash_attn3


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class _Case:
    """One backend in one configuration, and the answer the matrix gives it.

    `family` picks the KV-cache layout the phases are driven with: "mla" for
    the latent cache (`kv_lora_k_pe` and the two-query-tensor decode call) or
    "gqa" for a plain `k`/`v` paged cache.
    """

    name: str
    family: str
    in_graph: bool
    mtp_verify: bool
    build: Callable[[], AttnBackend]
    dtype: torch.dtype = torch.float16
    block_size: int = 0
    overlay: Optional[dict] = None
    skip: Optional[str] = None
    # The sparse MLA decode path: it attends only the indexer's TopK
    # positions, taken as a device tensor input, instead of the whole
    # context -- and with an fp8 cache it stores a packed fp8+scale row per
    # token. Both change what the harness has to build.
    sparse: bool = False
    packed_fp8: bool = False

    def __post_init__(self):
        if not self.block_size:
            object.__setattr__(
                self,
                "block_size",
                BLOCK_MLA if self.family == "mla" else BLOCK_GQA,
            )

    @property
    def pages_per_request(self) -> int:
        return KV_SPAN // self.block_size


def _config_for(case: _Case) -> dict:
    is_mla = case.family == "mla"
    models = {
        "n_heads": N_HEADS,
        "n_kv_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "dim": 7168,
        "type": "deepseek-v3" if is_mla else None,
    }
    if is_mla:
        models.update(
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        )
    config = {
        "infer": {
            "max_batch_size": BS,
            "use_cuda_graph": True,
            "tp_size": 1,
            "op_impl": "torch",
            "cache_type": "paged",
            "dp_size": 1,
            "max_seq_len": KV_SPAN,
            "mtp_size": MTP_SIZE,
            "mla_absorb": "absorb" if is_mla else "none",
        },
        "models": models,
    }
    return _merge(config, case.overlay or {})


def _prepare_case(case: _Case) -> None:
    """Put the config and the default dtype where construction reads them."""
    set_global_args(
        OmegaConf.create(_config_for(case)),
        need_ensure=False,
        need_preprocess=False,
    )
    torch.set_default_dtype(case.dtype)


def _flashinfer(family: str) -> FlashInferBackend:
    return FlashInferBackend(
        tot_num_blocks=BS * (KV_SPAN // (BLOCK_MLA if family == "mla" else BLOCK_GQA)),
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
    )


# One row per backend and per configuration that changes an answer, mirroring
# the support matrix in `chitu/attn_backend/README.md`. A row whose `skip` is
# set is one this machine cannot build; the flag test skips it with that reason
# rather than silently dropping it from the matrix.
_CASES = [
    _Case(
        name="ref_mla",
        family="mla",
        in_graph=False,
        mtp_verify=False,
        build=lambda: RefAttnBackend(qk_nope_head_dim=QK_NOPE_HEAD_DIM),
    ),
    _Case(
        name="ref_gqa",
        family="gqa",
        in_graph=False,
        mtp_verify=False,
        build=RefAttnBackend,
    ),
    _Case(
        name="npu_mla",
        family="mla",
        in_graph=False,
        # `cache_type != "paged"` would make this True; this case is paged.
        mtp_verify=False,
        build=lambda: NpuAttnBackend(qk_nope_head_dim=QK_NOPE_HEAD_DIM),
        skip=None if has_torch_npu else "torch_npu is missing",
    ),
    _Case(
        name="hunyuan_gqa",
        family="gqa",
        in_graph=False,
        mtp_verify=False,
        build=lambda: HunyuanAttnBackend(
            head_dim=HEAD_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS
        ),
        dtype=torch.bfloat16,
        overlay={"float_16bit_variant": "bfloat16"},
        skip=None if has_hunyuan_ops else "hunyuan_ops (hpc) is missing",
    ),
    _Case(
        name="dllm_gqa",
        family="gqa",
        in_graph=False,
        # Inherited from FlashAttnBackend and never consulted: dLLM replaces
        # `__call__`, so `route_to_decode` -- the only reader -- does not run.
        mtp_verify=True,
        build=DLLMAttnBackend,
        skip=None if _flash_attn_available() else "flash_attn is missing",
    ),
    _Case(
        name="triton_mla",
        family="mla",
        in_graph=not is_muxi(),
        mtp_verify=not is_muxi(),
        build=lambda: TritonAttnBackend(qk_nope_head_dim=QK_NOPE_HEAD_DIM),
        skip=None if has_triton_ops else "triton is missing",
    ),
    _Case(
        name="triton_gqa",
        family="gqa",
        in_graph=not is_muxi(),
        mtp_verify=not is_muxi(),
        build=TritonAttnBackend,
        skip=None if has_triton_ops else "triton is missing",
    ),
    _Case(
        name="flash_attn_mla",
        family="mla",
        in_graph=True,
        mtp_verify=True,
        build=lambda: FlashAttnBackend(qk_nope_head_dim=QK_NOPE_HEAD_DIM),
        # Without FA3 the MLA paged decode falls back to `_mla_to_mqa` and
        # FlashAttention's paged MQA kernel, which cannot take an MLA cache at
        # all: its head is `kv_lora_rank + qk_rope_head_dim` (576 here) and its
        # pages are 64-wide, against a 256 head limit and a 256-wide page on
        # CUDA. `test_attn.py` skips the same pair.
        skip=(
            None
            if has_flash_attn3
            else "MLA paged decode with flash_attn needs flash_attn_interface (FA3)"
        ),
    ),
    _Case(
        name="flash_attn_gqa",
        family="gqa",
        in_graph=True,
        mtp_verify=True,
        build=FlashAttnBackend,
        skip=None if _flash_attn_available() else "flash_attn is missing",
    ),
    _Case(
        name="hybrid_gqa",
        family="gqa",
        in_graph=_flash_attn_available(),
        mtp_verify=_flash_attn_available(),
        build=HybridAttnBackend,
    ),
    _Case(
        name="flash_infer_mla",
        family="mla",
        in_graph=True,
        mtp_verify=True,
        build=lambda: _flashinfer("mla"),
        skip=None if has_flashinfer else "flashinfer is missing",
    ),
    _Case(
        name="flash_infer_dense",
        family="gqa",
        in_graph=False,
        mtp_verify=False,
        build=lambda: _flashinfer("gqa"),
        skip=None if has_flashinfer else "flashinfer is missing",
    ),
    _Case(
        name="flash_mla_dense",
        family="mla",
        in_graph=_flash_mla_in_graph(),
        mtp_verify=True,
        build=lambda: FlashMLABackend(qk_nope_head_dim=QK_NOPE_HEAD_DIM),
        dtype=torch.bfloat16,
        skip=None if has_flash_mla else "flash_mla is missing",
    ),
    _Case(
        name="flash_mla_dense_e5m2",
        family="mla",
        in_graph=False,
        mtp_verify=False,
        build=lambda: FlashMLABackend(qk_nope_head_dim=QK_NOPE_HEAD_DIM),
        dtype=torch.bfloat16,
        overlay={
            "infer": {"mtp_size": 1},
            "models": {
                "quant_config": {
                    "kv_cache": {"rules": [{"regex": "kv_lora", "type": "fp8_e5m2"}]}
                }
            },
        },
        skip=(
            None
            if (is_hygon() and has_flash_mla and has_flash_mla_e5m2)
            else "fp8_e5m2 FlashMLA is Hygon-only"
        ),
    ),
    _Case(
        name="flash_mla_sparse_bf16",
        family="mla",
        in_graph=_flash_mla_in_graph(),
        mtp_verify=True,
        sparse=True,
        build=lambda: FlashMLABackend(
            qk_nope_head_dim=QK_NOPE_HEAD_DIM, index_topk=2048
        ),
        dtype=torch.bfloat16,
        overlay={"models": {"index_topk": 2048}},
        skip=None if has_flash_mla else "flash_mla is missing",
    ),
    _Case(
        name="flash_mla_sparse_fp8",
        family="mla",
        in_graph=_flash_mla_in_graph(),
        mtp_verify=True,
        sparse=True,
        packed_fp8=True,
        build=lambda: FlashMLABackend(
            qk_nope_head_dim=QK_NOPE_HEAD_DIM, index_topk=2048, use_fp8=True
        ),
        dtype=torch.bfloat16,
        overlay={"models": {"index_topk": 2048}},
        # The triton DSA quantizer supplies the packed-fp8 scale rows, and
        # Muxi's flashmla package has no FP8 kernel: building the backend there
        # raises before the flags can be checked.
        skip=(
            None
            if (has_flash_mla and has_triton_ops and not is_muxi())
            else "flash_mla or the triton DSA quantizer is missing, or no FP8 kernel"
        ),
    ),
    _Case(
        name="hopper_mixed",
        family="mla",
        in_graph=_flash_mla_in_graph(),
        mtp_verify=True,
        sparse=True,
        # Its FA3 sparse path reads one page per token.
        block_size=1,
        build=lambda: HopperMixedBackend(
            qk_nope_head_dim=QK_NOPE_HEAD_DIM, index_topk=2048
        ),
        dtype=torch.bfloat16,
        overlay={"models": {"index_topk": 2048}},
        skip=None if has_flash_attn3 else "flash_attn_interface (FA3) is missing",
    ),
]

_CASE_BY_NAME = {case.name: case for case in _CASES}
_CASE_IDS = [case.name for case in _CASES]
_IN_GRAPH_CASES = [case for case in _CASES if case.in_graph]

# Every backend class the driver may select, for the structural checks. The
# support matrix has a row for each of these.
_BACKEND_CLASSES = (
    AttnBackend,
    FlashAttnBackend,
    FlashInferBackend,
    FlashMLABackend,
    HopperMixedBackend,
    HunyuanAttnBackend,
    HybridAttnBackend,
    NpuAttnBackend,
    RefAttnBackend,
    TritonAttnBackend,
    DLLMAttnBackend,
)


class _Harness:
    """One case's step: its phases, its deltas and its KV cache.

    The verify phase feeds `MTP_SIZE` query tokens per sequence over the KV it
    appends, and each draft step after it feeds one more -- the step the MTP
    cache walks through, one position at a time, inside one capture.
    """

    def __init__(self, case: _Case):
        self.case = case
        self.family = case.family
        self.block = case.block_size
        _prepare_case(case)

        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(20260923)
        pages = BS * case.pages_per_request

        self.page_table = torch.arange(pages, dtype=torch.int32, device=device).view(
            BS, case.pages_per_request
        )

        if self.family == "mla":
            latent = _rand(generator, pages, self.block, KV_DIM)
            if case.packed_fp8:
                # The fp8 MLA cache holds one packed fp8+scale row per
                # token, which is what the real quantizer produces.
                latent = quant_pertoken_kvcache_dsa(latent)
            self.pristine = {"kv_lora_k_pe": latent}
            self.verify_q = (
                _rand(generator, BS * MTP_SIZE, N_HEADS, KV_LORA_RANK),
                _rand(generator, BS * MTP_SIZE, N_HEADS, QK_ROPE_HEAD_DIM),
            )
            self.draft_q = (
                _rand(generator, BS, N_HEADS, KV_LORA_RANK),
                _rand(generator, BS, N_HEADS, QK_ROPE_HEAD_DIM),
            )
            # This step's own tokens: `MTP_SIZE` per sequence on the verify
            # phase and one per draft step.
            self.verify_kv = _rand(generator, BS * MTP_SIZE, 1, KV_DIM)
            self.draft_kv = _rand(generator, BS, 1, KV_DIM)
        else:
            self.pristine = {
                "k": _rand(generator, pages, self.block, N_KV_HEADS, HEAD_DIM),
                "v": _rand(generator, pages, self.block, N_KV_HEADS, HEAD_DIM),
            }
            self.verify_q = _rand(generator, BS * MTP_SIZE, N_HEADS, HEAD_DIM)
            self.draft_q = _rand(generator, BS, N_HEADS, HEAD_DIM)
            self.verify_kv = (
                _rand(generator, BS * MTP_SIZE, N_KV_HEADS, HEAD_DIM),
                _rand(generator, BS * MTP_SIZE, N_KV_HEADS, HEAD_DIM),
            )
            self.draft_kv = (
                _rand(generator, BS, N_KV_HEADS, HEAD_DIM),
                _rand(generator, BS, N_KV_HEADS, HEAD_DIM),
            )

        shared = dict(
            device=device,
            max_batch_size=BS,
            max_total_len=KV_SPAN * BS,
            use_prefix_lens_static_tensor=False,
        )
        self.verify_delta = BatchedSeqLenDelta(
            [EARLY - MTP_SIZE] * BS,
            [EARLY] * BS,
            max_total_delta_len=BS * MTP_SIZE,
            **shared,
        )
        self.draft_delta = BatchedSeqLenDelta(
            [EARLY] * BS,
            [EARLY + 1] * BS,
            max_total_delta_len=BS,
            **shared,
        )

    def backend(self) -> AttnBackend:
        """One decode backend of this case."""
        return self.case.build()

    def blank(self) -> dict:
        """A KV cache that no phase of this step has written to yet."""
        return {name: tensor.clone() for name, tensor in self.pristine.items()}

    def accessor(self, kv: dict) -> PagedKVCacheAccessor:
        return PagedKVCacheAccessor(self.page_table, kv)

    def set_lens(self, start: int) -> None:
        """Rewind the step to the position `start`, outside any capture.

        A run advances the draft delta in place and appends this step's KV, so
        both the lengths and the cache have to be put back before the next one
        -- the model rewinds the same way, out of the graph, before every
        replay.
        """
        self.verify_delta.copy_from_list([start - MTP_SIZE] * BS, [start] * BS)
        self.draft_delta.copy_from_list([start] * BS, [start + 1] * BS)

    def rewind(self, start: int, kv: dict) -> None:
        """Put both the lengths and the cache back to where a replay starts."""
        self.set_lens(start)
        for name, tensor in self.pristine.items():
            kv[name].copy_(tensor)

    def reserve(self, backend: AttnBackend) -> None:
        """Declare every shape this step's capture will use, from outside it."""
        backend.reserve_metadata_for_decode(
            self.verify_delta, self.page_table, self.block
        )
        backend.reserve_metadata_for_decode(
            self.draft_delta, self.page_table, self.block
        )

    def topk_indices(self, delta: BatchedSeqLenDelta) -> torch.Tensor:
        """This phase's sparse indices, from the device lengths alone.

        Production gets them from the indexer, which derives them inside the
        same captured region the same way; all that matters here is that they
        follow the phase's device lengths rather than the recording's, and
        that every index is in range (the newest `TOPK` positions, clamped at
        the start of the context).
        """
        lens = delta.new.lens_tensor_device[delta.delta_seq_ids_tensor_device]
        offsets = torch.arange(TOPK, device=lens.device, dtype=torch.int32)
        return torch.clamp(lens[:, None] - 1 - offsets[None, :], min=0)

    def _phase(self, backend, delta, accessor, q, kv):
        backend.prepare_metadata_for_decode(delta, self.page_table, self.block)
        if self.family == "mla":
            if self.case.sparse:
                return backend.mla_decode_paged_kv(
                    *q, accessor, kv, delta, topk_indices=self.topk_indices(delta)
                )
            return backend.mla_decode_paged_kv(*q, accessor, kv, delta)
        return backend.decode_paged_kv(q, accessor, kv[0], kv[1], seq_len_delta=delta)

    def verify_phase(self, backend: AttnBackend, accessor) -> torch.Tensor:
        return self._phase(
            backend, self.verify_delta, accessor, self.verify_q, self.verify_kv
        )

    def draft_phase(self, backend: AttnBackend, accessor) -> torch.Tensor:
        return self._phase(
            backend, self.draft_delta, accessor, self.draft_q, self.draft_kv
        )

    def draft_phases(self, backend: AttnBackend, accessor) -> list[torch.Tensor]:
        """The step's draft phases, in the order the captured loop runs them."""
        outs = []
        for index in range(MTP_SIZE - 1):
            outs.append(self.draft_phase(backend, accessor))
            if index + 1 < MTP_SIZE - 1:
                self.draft_delta.advance_classic_by_one()
        return outs

    def phases(self, backend: AttnBackend, accessor) -> list[torch.Tensor]:
        """The step's phases, in the order the captured region runs them."""
        return [self.verify_phase(backend, accessor)] + self.draft_phases(
            backend, accessor
        )


def _capture(rewind, run):
    """Capture one call of `run`, with `rewind` putting the step at its start.

    The warmup runs on a side stream with the same rewind before each call, and
    the capture itself starts right after one, because `run` advances the step:
    a replay enters the region from the same state the recording did.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with _warmup():
            for _ in range(3):
                rewind()
                run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    rewind()
    # `_announce` covers a capture that traces a `cuda_graph_safe_cached_property`
    # value: the post-capture fix-up for it is registered on the graph object
    # being captured, which the model's own captures announce the same way (see
    # `make_dispatched_graphed_callables`).
    with _announce(graph):
        with torch.cuda.graph(graph):
            out = run()
    return graph, out


@contextlib.contextmanager
def _warmup():
    """Say that this call is the warmup that precedes a capture.

    `chitu.cuda_graph.is_warming_up_or_cuda_graph_capture` reads this flag, and
    some ops pick a different implementation under it -- the fp8 DSA quantizer
    selects its decode-graph kernel -- so a warmup that did not announce itself
    would leave that kernel to be autotuned for the first time *inside* the
    capture, which CUDA forbids.
    """
    assert chitu_cuda_graph._is_warming_up_before_cuda_graph_capture is False
    chitu_cuda_graph._is_warming_up_before_cuda_graph_capture = True
    try:
        yield
    finally:
        chitu_cuda_graph._is_warming_up_before_cuda_graph_capture = False


@contextlib.contextmanager
def _announce(graph: torch.cuda.CUDAGraph):
    """Say which graph object a hand-rolled `torch.cuda.graph` capture is for."""
    assert chitu_cuda_graph._currently_capturing_graph_object is None
    chitu_cuda_graph._currently_capturing_graph_object = graph
    try:
        yield graph
    finally:
        chitu_cuda_graph._currently_capturing_graph_object = None


def _assert_phases_match(actual, expected, where: str) -> None:
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected)):
        try:
            assert_close(got, want, atol=1e-2, rtol=1e-2)
        except AssertionError as error:
            raise AssertionError(f"{where}, phase {index}: {error}") from error


def _skip_if_unbuildable(case: _Case) -> None:
    if case.skip:
        pytest.skip(case.skip)


def _skip_unless_flashinfer_mla(case: _Case) -> None:
    _skip_if_unbuildable(case)
    if packaging.version.parse(flashinfer.__version__) < packaging.version.parse(
        "0.2.0"
    ):
        pytest.skip("flashinfer is too old")
    try:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper  # noqa: F401
    except ImportError:
        pytest.skip("flashinfer lacks the MLA wrappers")


def _overriders(method: str) -> set[str]:
    """The backend classes that replace `AttnBackend.<method>`."""
    base = getattr(AttnBackend, method)
    return {
        cls.__name__ for cls in _BACKEND_CLASSES if getattr(cls, method) is not base
    }


def test_reserve_metadata_for_prefill_is_the_base_no_op_for_every_backend():
    """No backend has anything to reserve for a prefill step.

    A prefill step is `reserve_metadata_for_prefill` ->
    `prepare_metadata_for_prefill` -> compute, and every backend sizes its
    prefill buffers on the fly, so the reservation stays the base no-op for all
    of them (see the contract table in `chitu/attn_backend/README.md`). This is
    a statement about the classes, so it needs no accelerator.
    """
    assert _overriders("reserve_metadata_for_prefill") == set()


def test_only_the_backends_with_something_to_do_replace_a_decode_hook():
    """Which decode hooks a backend replaces at all, class by class.

    The reservation is the base no-op for every backend but the one whose plan
    comes from a host-side scheduler, and the prepare is the base no-op for
    every backend whose metadata already comes from device tensors. A backend
    that grew a host-touching prepare would show up here, so it cannot hide
    behind a flag that still answers True.
    """
    assert _overriders("reserve_metadata_for_decode") == {"FlashInferBackend"}
    assert _overriders("prepare_metadata_for_decode") == {
        "FlashInferBackend",
        "FlashMLABackend",
        "HopperMixedBackend",
        "NpuAttnBackend",
    }


@needs_accelerator
@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_decode_prepare_flags_match_the_documented_answer(case: _Case):
    """Both flags of every case, against the documented support matrix."""
    _skip_if_unbuildable(case)
    _prepare_case(case)
    backend = case.build()
    assert backend.decode_supports_prepare_in_graph() == case.in_graph, (
        f"{case.name}: decode_supports_prepare_in_graph() disagrees with the "
        "matrix in chitu/attn_backend/README.md"
    )
    assert backend.decode_op_supports_mtp() == case.mtp_verify, (
        f"{case.name}: decode_op_supports_mtp() disagrees with the matrix in "
        "chitu/attn_backend/README.md"
    )


@needs_accelerator
@pytest.mark.parametrize(
    "case", _IN_GRAPH_CASES, ids=[case.name for case in _IN_GRAPH_CASES]
)
def test_one_reserve_then_every_phase_in_one_capture(case: _Case):
    """The promise itself: one reservation, every phase prepared in one capture.

    Reserve the verify and the draft shape outside the capture, run the step
    eagerly to build the plans and to have something to compare against, then
    capture the verify phase and both draft steps -- each preparing its own
    metadata, with the draft loop advancing the delta's device lengths by one
    between them -- and replay the whole thing at two quite different
    positions. Every replay has to reproduce, phase by phase, the eager result
    for the lengths that phase runs at: the captured loop moves the delta on by
    one per phase, so a prepare that had baked the recording's lengths would
    score a later phase with an earlier phase's KV span.
    """
    _skip_if_unbuildable(case)
    assert case.mtp_verify, (
        f"{case.name} claims in-graph prepares, which the single-graph MTP draft "
        "needs a verify phase for, so this row of the matrix is inconsistent"
    )
    harness = _Harness(case)
    backend = harness.backend()
    assert (
        backend.decode_supports_prepare_in_graph()
    ), "the case table claims this backend's prepares may run in a capture"
    harness.reserve(backend)

    eager = {}
    for name, start in (("early", EARLY), ("late", LATE)):
        harness.set_lens(start)
        eager[name] = harness.phases(backend, harness.accessor(harness.blank()))
    assert not torch.equal(eager["early"][0], eager["late"][0]), (
        "the phases have to depend on the position, or replaying them over "
        "another one would prove nothing"
    )

    kv = harness.blank()
    graph, captured = _capture(
        lambda: harness.rewind(EARLY, kv),
        lambda: harness.phases(backend, harness.accessor(kv)),
    )

    # Back and forth, so neither length can pass by having been recorded.
    for start, expected in (
        (LATE, eager["late"]),
        (EARLY, eager["early"]),
        (LATE, eager["late"]),
    ):
        harness.rewind(start, kv)
        graph.replay()
        torch.cuda.synchronize()
        _assert_phases_match(captured, expected, f"{case.name} replayed at {start}")


@needs_accelerator
def test_flashinfer_mla_plans_a_reserved_shape_only_outside_a_capture():
    """Reserving is what makes the later prepares legal, and where a plan is built.

    `reserve_metadata_for_decode` may not run while capturing -- it declares a
    shape *for* a captured region -- and once a shape is reserved, the plan it
    needs can only be built on the host, so capturing the first prepare for it
    (before any eager one) has to fail loudly instead of baking a plan into the
    graph's private pool.
    """
    case = _CASE_BY_NAME["flash_infer_mla"]
    _skip_unless_flashinfer_mla(case)
    harness = _Harness(case)
    backend = harness.backend()

    # Both captures below stop at the call under test, before any work is
    # recorded, which is what torch warns about at the end of an empty capture.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(AssertionError, match="must be called outside one"):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                backend.reserve_metadata_for_decode(
                    harness.draft_delta, harness.page_table, harness.block
                )

    harness.reserve(backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(RuntimeError, match="outside a captured region"):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                backend.prepare_metadata_for_decode(
                    harness.draft_delta, harness.page_table, harness.block
                )


@needs_accelerator
def test_flashinfer_mla_reserved_drafts_match_the_host_planned_ones():
    """A reserved shape's metadata is a wider plan, not a different answer.

    The device-driven plan spans the whole reserved KV range and masks every
    phase with that phase's own device length, which is what makes one plan
    serve every replay. That only holds up if it scores the same as the plan
    the host builds from exactly this phase's page range.
    """
    case = _CASE_BY_NAME["flash_infer_mla"]
    _skip_unless_flashinfer_mla(case)
    harness = _Harness(case)
    reserved = harness.backend()
    host = harness.backend()
    harness.reserve(reserved)

    for start in (EARLY, LATE):
        harness.set_lens(start)
        kv_reserved = harness.blank()
        kv_host = harness.blank()
        # The verify shape cannot be host-planned -- its plan needs a query row
        # per token in the host buffers and there is one per sequence -- so both
        # sides take it from the reserved backend, and the draft phases are what
        # the two plans are compared on.
        harness.verify_phase(reserved, harness.accessor(kv_reserved))
        harness.verify_phase(reserved, harness.accessor(kv_host))

        device_drafts = harness.draft_phases(reserved, harness.accessor(kv_reserved))
        harness.set_lens(start)
        host_drafts = harness.draft_phases(host, harness.accessor(kv_host))
        _assert_phases_match(device_drafts, host_drafts, f"at {start}")
