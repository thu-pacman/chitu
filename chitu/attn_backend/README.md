# Attention backends: the call contract and the support matrix

`chitu/attn_backend/` holds one class per attention implementation. The class is
picked by `Backend._get_attention_backend_type` (`chitu/backend.py:540`) from
`infer.attn_type`, which `auto` resolves to
`npu` on Ascend, `ref` when `infer.op_impl=cpu`, `hopper_mixed` for bf16 sparse
MLA on Hopper, `flash_mla` for the DeepSeek / Kimi-K2.5 / GLM-5.x families, and
`hybrid` otherwise.

This file is the reference for how the driver calls a backend, and for which
backends may re-derive a decode step's metadata on the GPU instead of the host.
Keep it in sync when a backend's answer changes.

## The contract

Every step is `reserve` → `prepare` → compute:

| stage | prefill step | decode step |
| --- | --- | --- |
| reserve, outside any capture | `reserve_metadata_for_prefill` (`chitu/attn_backend/base.py:80`) | `reserve_metadata_for_decode` (`chitu/attn_backend/base.py:98`) |
| prepare | `prepare_metadata_for_prefill` (`chitu/attn_backend/base.py:130`) | `prepare_metadata_for_decode` (`chitu/attn_backend/base.py:123`), once per phase |
| compute | `prefill_*` entry points | `decode_*` / `mla_decode_*` entry points |

The four methods in `base.py` carry the full contract. In short, **reserve**
sizes from what the step *cannot exceed* -- the addressable KV span, the block
table's width -- rather than from its own lengths, because one decode
reservation covers every phase of its capture at once; **prepare** then fills
those buffers in from the step's *actual* lengths, which is legal inside a
capture exactly when `decode_supports_prepare_in_graph` answers True.

A decode step's *phases* are:

- a plain decode step (one query token per sequence), or, with MTP, one verify
  phase carrying `mtp_size` query tokens per sequence;
- plus, when the MTP draft loop is captured on its own, the `mtp_size - 1` draft
  phases, one query token per sequence each.

The driver (`chitu/models/model.py`) therefore does:

- prefill: `prefill()` → `reserve_prefilling_attn()` (`:2277`) →
  `prepare_metadata_for_prefill` (`:2263`);
- decode: `decode()` → `reserve_decoding_attn()` (`:2288`), then either one
  `prepare_decoding_attn()` (`:2319`) per phase outside the graph, or every
  phase's prepare inside the graph (`:2455`);
- MTP draft loop captured as one graph: `_prepare_mtp_draft_step1()` (`:1820`)
  reserves the draft shape with `phases_after=mtp_size - 2`, so that one
  reservation covers the `mtp_size - 1` phases the capture replays.

## `decode_supports_prepare_in_graph`

`chitu/attn_backend/base.py:61` states what answering True requires: the phase's
prepare reads device tensors only -- no host mirror (`lens_list`,
`lens_tensor_cpu`, `max_len`, `delta_max_len`), no device-to-host sync -- and any
plan the kernel needs is either length-independent or rebuilt by the kernel from
the device lengths on every replay. What it buys is that a capture may hold every
phase's prepare, not just the first, which is what lets one replay produce all
K-1 draft tokens.

Answering False costs one captured graph per phase: the driver keeps
`prepare_decoding_attn` in `before_capture_callback`
(`chitu/models/model.py:2447`), and `_use_mtp_draft_single_graph` (`:1661`)
declines to capture the draft loop at all, degrading to `_draft_eager` with a
warning.

`decode_op_supports_mtp` (`chitu/attn_backend/base.py:58`) is a separate
question: whether the decode op accepts a verify phase, i.e. `mtp_size` query
tokens per sequence in one call. A backend that says no routes a verify step
through its prefill-shaped entry point (`route_to_decode`, `:241`), which is
typically host-shaped and therefore also keeps `decode_supports_prepare_in_graph`
False.

## Support matrix

`attn_type` is the value of `infer.attn_type` (or what `auto` resolves to).
"MTP verify" is `decode_op_supports_mtp`; "in-graph dec. prepare" is
`decode_supports_prepare_in_graph`.

| backend | `attn_type` | MTP verify | in-graph dec. prepare | answer made in |
| --- | --- | --- | --- | --- |
| FlashAttention | `flash_attn`, and the `hybrid` sub-backend | yes | yes | `flash_attn_backend.py:53` |
| Triton | `triton` | yes, except Muxi | yes, except Muxi | `triton_attn_backend.py:179` |
| FlashMLA | `flash_mla`, `auto` for DSV3 / Kimi-K2.5 / GLM-5.2 / GLM-5-Next | yes, except an e5m2 cache | Nvidia yes; Hygon yes with `FlashMLASchedMeta`, no with the old ABI; Muxi no | `flash_mla_backend.py:312` |
| FA3 qv-split MLA | `hopper_mixed` | inherited | inherited | inherits `flash_mla_backend.py:312` |
| FlashInfer | `flash_infer` | MLA only | paged MLA only | `flash_infer_backend.py:394` |
| hybrid | what `auto` falls back to; no separate name | yes | yes, via its FlashAttention sub-backend | `hybrid_attn_backend.py:64` |
| Hunyuan | `hunyuan_attn` | no | no | `hunyuan_attn_backend.py:107` |
| Ascend / NPU | `npu`, and `auto` on Ascend | yes unless `cache_type=paged` | no | `npu_attn_backend.py:139` |
| dLLM | `dllm` | yes, inherited and unused | no | `dllm_backend.py:41` |
| reference / CPU | `ref`, `auto` with `op_impl=cpu` | no | no | `ref_attn_backend.py:30` |

`test/pytest/test_attn_decode_capture.py` is the executable form of this table:
one case per row and per configuration that changes an answer (a KV layout, an
e5m2 or fp8 cache, a sparse indexer), each checked against the answer above, and
every case that claims an in-graph prepare is driven through one
`reserve_metadata_for_decode` outside a capture and a whole MTP step -- verify
phase plus both draft steps -- inside one, replayed at two quite different
positions. Cases a machine cannot build are skipped by name, so the table is
stated everywhere and verified as far as the platform reaches.
`test/pytest/test_model_decode_capture_decision.py` covers the other half: how
the driver turns these two flags into the choice between one captured draft loop
and a per-step draft, including the negative paths -- a backend that answers
False must keep the loop on the host.

## Why each backend answers what it does

- **FlashAttention** (`flash_attn_backend.py:49`): the decode entry points take
  everything that varies between two draft steps as a device tensor
  (`cache_seqlens`, `block_table`) and size their scratch from fixed
  batch/head shapes. The backend overrides no decode prepare, so the base no-op
  is the whole preparation, and the MLA path is device-only too (FA3 MLA, or
  `_mla_to_mqa` → `decode_paged_kv`).
- **Triton** (`triton_attn_backend.py:167`, `:179`): the same no-op prepare, and
  the kernels launch from `seq_len_delta.*.lens_tensor_device` and the block
  table, with a `num_kv_splits` that does not follow the per-step lengths. On
  Muxi the MLA decode falls back to MQA and `num_kv_splits` comes from the batch
  size, so both answers are False there.
- **FlashMLA** (`flash_mla_backend.py:312`): a decode prepare is capture-safe
  exactly when the plan it hands the kernel is *not* a plan of that phase's
  lengths. The `FlashMLASchedMeta` interface
  (`chitu/attn_backend/flash_mla_backend.py:30`) returns an empty plan from
  `get_mla_metadata()` and lets the first `flash_mla_with_kvcache` fill it from
  device `cache_seqlens` on each replay, which is capture-safe; the older
  interface plans on the host from those seqlens, which would get baked. Nvidia
  always ships the new interface, Hygon ships it in recent packages (gated by
  `has_flash_mla_sched_meta`, a package capability rather than a platform), and
  Muxi has not been checked either way. An e5m2 cache keeps the classic
  single-token kernel, so it also answers False for an MTP verify.
- **FlashInfer** (`flash_infer_backend.py:404`, `:422`): the only backend that
  overrides `reserve_metadata_for_decode`, because its MLA plan comes from a
  host-side scheduler. A reservation registers the shape, the first prepare for
  that shape builds the plan outside the capture, and every later prepare for
  the same shape only rewrites the step's page table and KV lengths from the
  device (`_MlaGpuInputPlan`). The non-MLA wrapper stays a host-planned
  single-token kernel, and the dense fallback does not accept a verify phase.
- **hybrid** (`hybrid_attn_backend.py:60`): MTP decode always short-circuits to
  the FlashAttention sub-backend, so the parent's answer is the child's, and
  False when no child was built.
- **Hunyuan** (`hunyuan_attn_backend.py:107`): no decode metadata to prepare at
  all, but the decode op does not take a verify phase, and the prefill-shaped
  entry point that a verify step lands on sizes its query extent and mask from
  the host `delta_max_len` mirror.
- **Ascend** (`npu_attn_backend.py:139`): the Ascend operator takes its KV
  lengths as a host list (`actual_seq_lengths_kv`) and the decode prepare sizes
  its mask from `new.max_len` / `delta_max_len`. A capture feeds the refreshed
  host list through `cpu_update_input` instead, so the preparation stays outside
  the graph.
- **dLLM** (`dllm_backend.py:41`): block-wise bidirectional decoding whose
  per-block metadata is written from the host in `prepare_decode` /
  `init_static_tensors_for_decode`; it opts out explicitly even though its
  parent, `FlashAttnBackend`, opts in. Its `decode_op_supports_mtp` answer is
  the parent's, and is never read: dLLM replaces `__call__`, so the
  `route_to_decode` that would consult it does not run.
- **reference** (`ref_attn_backend.py:30`): the readable host-side
  implementation, walking `old.lens_list` / `new.lens_list` and sizing dense K/V
  from `new.max_len` in Python.

## DSA indexer

Models whose config sets `index_topk` also carry an indexer, chosen by
`infer.indexer_type` and resolved by `chitu/global_vars.py:314`. It scores the
cached indexer-K against the step's query and hands the sparse MLA the TopK
positions to attend to. It mirrors the attention contract with one difference:
its `reserve_metadata_for_decode`
(`chitu/dsa_indexer_backend/base.py:75`) takes `phases_after` explicitly, and its
`decode_supports_prepare_in_graph` (`:90`) is asked separately, so a model can
have a capture-ready attention backend and a host-bound indexer, or the other
way round. `Transformer._decode_prepare_in_graph` (`chitu/models/model.py:1705`)
requires both.

| indexer | `indexer_type` | in-graph dec. prepare | answer made in |
| --- | --- | --- | --- |
| BF16 torch | `torch_bf16` | yes | `chitu/dsa_indexer_backend/base.py:109` |
| BF16 triton | `triton_bf16` | yes | same |
| FP8 triton | `triton` | yes | same |
| FP8 DeepGEMM | `deepgemm` | yes | same |
| Hygon | `hygon` | yes | same, with `reserve_metadata_for_decode` at `hygon_backend.py:409` |
| FP8 torch | `torch` | no | same |

- Captured indexer phases were checked on hardware rather than inferred: on a
  real HCU, a decode phase captured together with its score and top-k
  reproduces the eager logits and indices for every replayed length set, under
  both a full verify row count and a single draft row.
- Every backend that qualifies scores from device tensors: the FP8 `triton`
  score loads `delta_seq_ids` / `delta_position_ids` / `new.lens_tensor_device`
  and the page table (`chitu/ops/triton_ops/quant/blockfp8/index_score.py`), the
  BF16 `torch_bf16` one masks with `new.lens_tensor_device`
  (`chitu/dsa_indexer_backend/torch_backend.py`), and `triton_bf16` runs the
  same shape through a triton kernel (`triton_backend.py:47`).
- `deepgemm` schedules its paged MQA logits with
  `deep_gemm.get_paged_mqa_logits_metadata(lens_tensor_device, 64, num_sms)`
  (`chitu/dsa_indexer_backend/deepgemm_backend.py:56`): the host side only reads
  dims and emits a length-independent `[num_sms + 1, 2]` buffer, the scheduling
  runs on the device, and the score rows are sized by the static
  `static_max_n`. `test/pytest/test_indexer_decode_capture.py` replays it over
  growing device lengths.
- `hygon` works the same way through its own DeepGEMM package, except for one
  bound: its fused decode TopK derives each row's split count from the device
  lengths and takes one plan for the whole capture, so
  `reserve_metadata_for_decode` pins that bound from the host lengths, over all
  the phases the capture replays (`hygon_backend.py:392`). A lower bound stays
  correct -- the rows it misses fall back to the exact P1 selector -- but gives
  up their split.
- The FP8 `torch` impl is a real blocker, not a missing measurement: its paged
  score
  (`blockfp8_index_score_ragged_q_paged_k_dsv32_torch`,
  `chitu/ops/quant/blockfp8/index_score.py`) scatters the paged cache into a
  dense `[b, static_max_n]` K using the *K-side* ids
  `new.seq_ids_tensor_device` / `new.position_ids_tensor_device`, both derived
  from the host mirror (`total_len` ← `lens_tensor_cpu` ← `lens_list`,
  `chitu/batched_seq_len.py`). A captured draft step advances the device lengths
  without that mirror and a read then raises "host sequence lengths are stale";
  and even with a fresh mirror their length is the whole context, which grows
  every step, so a capture bakes it and a replay at a longer context scores a
  *truncated* context -- the newest tokens silently stop being selectable.
  Admitting it needs the score to read device ids, not a new probe.

## Adding or changing a backend

1. Implement the two flag methods with the reason in a comment next to them --
   this file lists the ones that are False.
2. If the decode prepare needs buffers, size them in
   `reserve_metadata_for_decode` from allocated sizes, and assume it covers
   `phases_after + 1` phases.
3. Claim `decode_supports_prepare_in_graph` only after checking that no read of
   `lens_list` / `*_cpu` / `max_len` / `delta_max_len` and no device-to-host sync
   happens on the path, e.g. by capturing a phase and replaying it at a longer
   length than it was captured at.