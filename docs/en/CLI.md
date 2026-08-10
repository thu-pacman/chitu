# Chitu CLI Arguments

This document is automatically generated from `chitu/config/serve_config.yaml` by `script/generate_cli_docs.py`. Do not edit it by hand.

## `defaults`

`defaults` are other config files or scripts included into this config file. See
https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/ for details.

### Argument `defaults.models`

Include model configs into this file. Please set it to the model you want to run.

Acceptable values: One of the model names listed in `docs/en/SUPPORTED_MODELS.md`,
which is essentially one of the file names without suffix in `chitu/config/models`.

E.g., `Qwen2-7B-Instruct` means including `chitu/config/models/Qwen2-7B-Instruct.yaml`
into this file.

IMPORTANT: You always need to set this field.

*Default: `???`.*

## `boot`

Configs for how chitu get to run on multiple servers or multiple GPUs

### Argument `boot.n_nodes`

Number of nodes (servers) to use

*Default: `1`.*

### Argument `boot.ssh_node_list`

List of nodes (servers) to use. Please fill with hostnames or IPs.

This field is only for, and must only be set for `boot.remote_launcher=ssh`.
If set, the length of this list must equal to `n_nodes`.

E.g., ["node1", "node2"]

*Default: `null`.*

### Argument `boot.n_gpus_per_node`

Number of GPUs per node to use

*Default: `1`.*

### Argument `boot.job_name`

Slurm job name and Docker container name prefix.

Acceptable values:
- null: Use `${USER}-chitu` as the Slurm job name, and do not set Docker container names.
- A string: Use this value as the Slurm job name and Docker container name prefix.

*Default: `null`.*

### Argument `boot.container_image`

Path to container image (override bundled image).

For apptainer: absolute path to a .sif file.
For docker: image name (e.g. "chitu-ci-build:main").
When null, uses the image bundled in the AppImage.

*Default: `null`.*

### Argument `boot.platform`

Hardware platform used to configure device access for the container

Acceptable values:
- "auto": Detect the platform on each node from the available system management
  command.
- "nvidia": Use NVIDIA container arguments without checking for `nvidia-smi`.
- "ascend": Use Ascend container arguments without checking for `npu-smi`
  (Docker only).
- "hygon": Use Hygon container arguments without checking for `hy-smi`.
- "metax": Use MetaX container arguments without checking for `mx-smi`
  (Docker only).

Auto detection checks NVIDIA, Ascend, Hygon, and MetaX in that order for Docker.
Apptainer supports and checks only NVIDIA and Hygon.

*Default: `"auto"`.*

### Argument `boot.source_path`

Path to source tree to mount into the container at /workspace/chitu.

When set, the directory is bind-mounted into the container, so code changes
take effect without rebuilding the image. Requires the source tree to be
accessible at the same path on all nodes (e.g. on NFS).
When null, uses the code baked into the container image.

*Default: `null`.*

### Argument `boot.target`

Target chitu program or script

Acceptable values:
- A list of arguments.
- A space-separated string for arguments.

*Default: `["-m", "chitu"]`.*

### Argument `boot.relay_args`

Relay Hydra arguments from chitu-boot to inner chitu service

Acceptable values:
- True: Usually you choose this, to make chitu-boot and chitu service consistent.
- False: Choose this if you want to explicitly pass arguments to inner chitu service in
  `boot.target`, or if you want to boot non-chitu programs.

*Default: `True`.*

### Argument `boot.on_ready`

Invoke this command once chitu service starts

This command will run alongside Rank 0 for single-instance deployment, or run alongside
the router for multi-instance deployment. The process will run on a dedicated container,
started with the same argument as the chitu service.

Acceptable values:
- null: No command will run.
- A list of arguments.
- A space-separated string for arguments.

*Default: `null`.*

### Argument `boot.on_ready_relay_args`

Relay Hydra arguments from chitu-boot to the command in `boot.on_ready`

Acceptable values: True or False.

*Default: `True`.*

### Argument `boot.on_ready_shutdown`

Terminate chitu service once the command in `boot.on_ready` finishes (either successfully
or not)

*Default: `False`.*

### Argument `boot.remote_launcher`

How to run on multiple nodes

Acceptable values:
- local: Only run on the current node.
- srun: Use `srun` from Slurm.
- ssh: Directly use SSH to connect to remote nodes.

*Default: `"local"`.*

### Argument `boot.interactive_node_0`

Make the first node interactive, but output from the other nodes will NOT be printed
onto the terminal.

Acceptable values:
- "auto": Decide automatically.
- True: Make the first node interactive, but output from the other nodes will NOT be
  printed onto the terminal.
- False: Do not make the first node interactive, but output from all nodes will be
  printed onto the terminal.

*Default: `"auto"`.*

### Argument `boot.ascend_only_mount_visible_dev`

Only mount devices selected by `ASCEND_RT_VISIBLE_DEVICES` to the container

Acceptable values:
- True: Only mount devices selected by `ASCEND_RT_VISIBLE_DEVICES`.
- False: Mount all devices detected as `/dev/davinci<npu_id>`.

*Default: `False`.*

### Argument `boot.extra_srun_args`

Additional arguments passed to `srun`

Acceptable values:
- A list of arguments.
- A space-separated string for arguments.

*Default: `[]`.*

### Argument `boot.extra_apptainer_args`

Additional arguments passed to `apptainer run`

Acceptable values:
- A list of arguments.
- A space-separated string for arguments.

*Default: `[]`.*

### Argument `boot.extra_docker_args`

Additional arguments passed to `docker run`

By default, Chitu passes `--rm` to `docker run`, so containers are automatically
removed when they exit. To keep stopped containers, pass `--rm=false` in this field.

Acceptable values:
- A list of arguments.
- A space-separated string for arguments.

*Default: `[]`.*

### Argument `boot.extra_torchrun_args`

Additional arguments passed to `torchrun`

Acceptable values:
- A list of arguments.
- A space-separated string for arguments.

*Default: `[]`.*

### Argument `boot.container_setup_cmd`

Bash commands to set up each Docker or Apptainer workload container before launching its main
command

The commands run after the container starts and before `boot.torchrun_wrapper` and `torchrun`.
They run sequentially in the same non-interactive Bash shell. No shell options are changed and
no variables are exported automatically. Environment variables explicitly exported by the
commands are visible to `torchrun`. In multi-instance deployments, the commands run once in
each workload container, including the router. They do not run in the container for
`boot.on_ready`.
Commands may run concurrently across containers and nodes, so they must be safe to repeat and
must not modify the same shared files without coordination. Referenced files must be mounted
into the container separately. The command string may be visible in process listings and
startup logs, so do not put secrets directly in this setting.

Acceptable values:
- null: Run no extra commands.
- A string containing trusted Bash commands. Use a YAML block scalar (`|`) for multiple
  commands.

*Default: `null`.*

### Argument `boot.torchrun_wrapper`

Optional arguments to appear before `torchrun`

Acceptable values:
- A list of arguments.
- A space-separated string for arguments.

*Default: `[]`.*

## `models`

Configs for the model to inference.

### Argument `models.ckpt_dir`

Path to the model checkpoint directory.

Acceptable values: /path/to/your/checkpoint

Note to developers: We use `null` instead of `???` here because we have better error messages
for value missing in `chitu_main.py`, while missing values to a `???` field will always triggers
Hydra's default error message.

IMPORTANT: You always need to set this field.

*Default: `null`.*

### Argument `models.tokenizer_path`

Path to the tokenizer.

If null, the tokenizer will be loaded from `ckpt_dir`.

Acceptable values: null, /path/to/your/tokenizer

*Default: `null`.*

## `serve`

Configs for how chitu responses to HTTP requests.

### Argument `serve.host`

HTTP service IP. Set this according to your network.

*Default: `0.0.0.0`.*

### Argument `serve.port`

HTTP service port. Set this according to your network.

*Default: `21002`.*

### Argument `serve.api_keys`

If the request has a api_key field found in this dict, the request will be prioritized
according to the `priority` field. The higher the value of the field, the higher the
priority. Ordinary request has a priority of 1.

## `coordinator`

Config of a TCP store used for coordinating internal TCP connections.

This coordinator is responsible for managing TCP ports, but the address and port for itself
must be set here sometime.

The address and port can be omitted in the following cases:
- They can be omitted if `multi_inst.n_insts == 1`. If either `coordinator.host` or
  `coordinator.port` is null, `coordinator` will reuse key-value store from `torchrun`, and
  do not start a new key-value store.
- They can be omitted when launching from chitu.run (via `chitu.boot` module).

### Argument `coordinator.host`

IP for the coordinator.

Acceptable values:
- null: This field can be omitted in cases described above.
- A string: E.g., "1.2.3.4" or "host1". It must be recognized from all nodes,
  and therefore it must NOT be wildcards like 0.0.0.0. Please set concrete IP
  addresses.

*Default: `null`.*

### Argument `coordinator.port`

Port for the coordinator.

Acceptable values:
- null: This field can be omitted in cases described above.
- An integer: E.g., 21001. The TCP port ID.

*Default: `null`.*

## `infer`

Configs for how chitu do the LLM inference computation.

### Argument `infer.max_batch_size`

Maximum number of concurrent inference tasks (the actual batch size limit).
The actual batch size may be smaller due to memory limit, or insufficient
requests. If the number of concurrent requests exceeds this value, some
of the requests will wait.

This value is the global (total) value across DP ranks.

If not set, defaults to max_reqs for backward compatibility.

IMPORTANT: Typically, you need to set this field.

*Default: `8`.*

### Argument `infer.max_concurrent_requests`

Maximum concurrent requests (running + queued). When exceeded, new inference
requests are rejected with HTTP 503. If not set, defaults to max_batch_size * 2.

*Default: `null`.*

### Argument `infer.max_seq_len`

Hard sequence length limit of a single request, WITHOUT considering memory limit. This
is the maximum number of input and output tokens in total.

IMPORTANT: Typically, you need to set this field.

*Default: `10240`.*

### Argument `infer.cache_type`

Data structure for KV cache. Set it to "page" for paged KV cache

Acceptable values:
- "skew": For better performance when the memory is sufficient. This is a legacy name for
  dense KV cache.
- "paged": To better handling requests with different lengths.

IMPORTANT: Typically, you need to set this field.

*Default: `skew`.*

### Argument `infer.pcp_size`

Number of Prefill Context Parallel ranks. See `docs/en/DEVELOPMENT.md#parallelism`
for details.

*Default: `1`.*

### Argument `infer.tp_size`

Number of Tensor Parallel ranks for non-MoE models or non-MoE modules in MoE models.
See `docs/en/DEVELOPMENT.md#parallelism` for details.

*Default: `1`.*

### Argument `infer.pp_size`

Number of Pipeline Parallel ranks for non-MoE models or non-MoE modules in MoE models.
See `docs/en/DEVELOPMENT.md#parallelism` for details.

*Default: `1`.*

### Argument `infer.dp_size`

Number of Data Parallel ranks for non-MoE models or non-MoE modules in MoE models.
See `docs/en/DEVELOPMENT.md#parallelism` for details.

*Default: `1`.*

### Argument `infer.ep_size`

Number of Expert Parallel ranks for MoE modules in MoE models. See
`docs/en/DEVELOPMENT.md#parallelism` for details.

*Default: `1`.*

### Argument `infer.etp_size`

Number of Expert Parallel ranks for MoE modules in MoE models. See
`docs/en/DEVELOPMENT.md#parallelism` for details.

*Default: `null`.*

### Argument `infer.embed_tokens_lm_head_tp_size`

Number of Tensor Parallel ranks for embed_tokens and lm_head.

*Default: `auto`.*

### Argument `infer.seed`

Random seed

*Default: `0`.*

### Argument `infer.attn_type`

Attention backend.

Acceptable values:
- "auto": Automatically choose a good backend.
- "flash_attn": Use flash_attn. This requires installing chitu with `chitu[flash_attn]`
  for extra dependency.
- "flash_mla": Use flash_mla. This requires installing chitu with `chitu[flash_mla]`
  for extra dependency.
- "flash_infer": Use flashinfer. This requires installing chitu with `chitu[flashinfer]`
  for extra dependency.
- "triton": Use chitu's built-in triton backend. This requires running on a platform
  supporting Triton.
- "npu": Use operators dedicated for Ascend NPUs.
- "hopper_mixed": Special hybrid backend for running sparse attention on Hopper GPUs.
- "ref": Use chitu's built-in reference backend. This is backend has full support for
  different types of attention, but it is very slow and consumes much memory.

*Default: `auto`.*

### Argument `infer.indexer_type`

Indexer impl type, only valid for models with DSA (DeepSeek Sparse Attention).

Acceptable values:
- "auto": Automatically choose.
- "deepgemm": FP8 indexer KV cache path using deep_gemm mqa logits and a fused
  paged indexer-kv layout (requires deep_gemm).
- "triton": FP8 indexer KV cache path using Triton kernels and a separate
  (paged/skew) indexer-kv layout.
- "torch": FP8 indexer KV cache reference/fallback path sharing the same
  separate (paged/skew) indexer-kv layout as "triton".
- "hygon": BF16 indexer KV cache path for the Hygon platform.
- "torch_bf16": BF16 indexer KV cache pure-torch mqa logits path.
- "triton_bf16": BF16 indexer KV cache path using Triton mqa logits kernels,
  sharing the BF16 (K-only) indexer-kv layout with "torch_bf16"/"hygon".

FP8 indexer paths require a kv_cache rule matching indexer_k with
type=fp8_pertoken_indexer. BF16 indexer paths require unquantized indexer KV cache.

*Default: `auto`.*

### Argument `infer.op_impl`

Currently this option is only for enabling/disabling muxi_custom_kernel.

Acceptable values:
- "torch": Ordinary implementation.
- "muxi_custom_kernel": Use additional kernels for running on MetaX GPUs, optimized for
  small batches. This requires installing chitu with `chitu[muxi_custom_kernel]` for
  extra dependency.

*Default: `torch`.*

### Argument `infer.mla_absorb`

Absorption mode for MLA. This field is ignored when the model does not contain MLA.

Acceptable values:
- "auto": Decide automatically.
- "none": No absorption. This is an optimization for lower FLOP counts, which matches the
  typical need for prefilling.
- "absorb-without-precomp": exchange some matrices in the order of multiplication. This is
  an optimization for smaller memory footprint and lower memory occupancy for KV cache,
  which matches the typical need for decoding.
- "absorb-kv-only": keep the same latent-only KV cache and absorbed weight layout as
  "absorb-without-precomp", but reconstruct full K/V from kv_lora + k_pe during prefill
  before calling normal attention. This is intended for Prefill instances that want
  flash-attn over full K/V while preserving latent-only cache storage and PD transfer.
- "absorb": exchange some matrices in the order of multiplication, and precompute all the
  multiplications that can be computed before inference. This is an optimization for fewer
  operator counts, which may be useful for low-latency + low-concurrency cases.

*Default: `"auto"`.*

### Argument `infer.raise_lower_bit_float_to`

The hardware-supported data type used for software implementation for lower-bit data types
that are not supported by the hardware. For example, when you want to run a float8_e4m3fn
typed model with GPUs only supporting bfloat16 instructions, set this field to bfloat16.

*Default: `float8_e4m3fn`.*

### Argument `infer.fuse_shared_experts`

Whether to fuse shared experts and routed experts into the same operators. This field is
ignored for non-MoE models, or for MoE models with no shared experts.

When enabled, shared experts are treated as part of MoE blocks and partitioned by
`infer.etp_size`. When disabled, shared experts are treated as ordinary dense modules
and partitioned by `infer.tp_size`.

Acceptable values: True, False

*Default: `False`.*

### Argument `infer.device_ids`

If not null, override device IDs assgiend to each rank.

Please note that the ranks are global ranks, but the device IDs are local to node.
Therefore, if you want to use all 16 GPUs on two 8-GPU nodes, you can set
`device_ids: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]`.

Acceptable values:
- null: Each rank is assigned with the local-rank-id-th device.
- a list of integers: E.g, [2, 1, 3] means device 2, 1, 3 are assigned to ranks 0, 1, 2,
  respectively.

*Default: `null`.*

### Argument `infer.pp_layer_partition`

If not null, override the automatic layer partitioning for Pipeline Parallelism.

Acceptable values: null or a list of integers, e.g., [10, 12, 12, 10].

*Default: `null`.*

### Argument `infer.use_cuda_graph`

Whether to use CUDA graph or equivalent technologies on non-CUDA platforms.

Acceptable values:
- "auto": Decide automatically.
- True: Use CUDA graph.
- False: Do not use CUDA graph.

*Default: `auto`.*

### Argument `infer.minimax_sparse_decode_backend`

MiniMax M3 only: sparse-layer decode backend.
- "remap": gather selected sparse blocks and use the configured dense attention backend (default)
- "triton": per-KV-head block sparse attention

*Default: `remap`.*

### Argument `infer.minimax_sparse_prefill_backend`

MiniMax M3 only: sparse-layer prefill backend.
- "auto": Triton block sparse when max prefill query len >= 9216 (H20-tuned), else dense Flash
- "dense_flash": always use dense Flash attention (legacy fallback)
- "triton": always use Triton block sparse prefill

*Default: `auto`.*

### Argument `infer.memory_utilization`

Device memory utilization rate for automatic page allocation for paged KV cache.

Chitu will try to allocate as many pages as possible, satisfying that
`kv_cache_mem + weight_mem + estimated_activation_mem <= memory_utilization * total_gpu_mem`,
where `estimated_activation_mem` is estimated during engine warmup. Since the
esitimation may not be accurate enough, `memory_utilization` may be needed to be
tuned.

Acceptable values: 0.0 to 1.0, e.g., 0.98 means 98% of GPU memory will be used.

*Default: `0.98`.*

### Argument `infer.num_blocks`

If not -1, override the automatic page allocation for paged KV cache, and disables the
`memory_utilization` field.

Acceptable values: -1 or a positive integer.

*Default: `-1`.*

### Argument `infer.prefill_chunk_size`

Prefill chunk size. A higher value will increase prefilling throughput, but also increase
memory usage for intermediate tensors.

Acceptable values:
- "auto": Decide automatically.
- null: Disable prefill chunking.
- a positive integer: The global (total) prefill chunk size across DP ranks.

*Default: `auto`.*

### Argument `infer.mtp_size`

Number of total tokens generated by the main model and then MTP layers in a single step,
which means 1 token is generated by the main model and (mtp_size - 1) tokens are generated
by MTP layers. Setting to 1 means disabling MTP. This field is ignored when the model does
not support MTP.

This value needs to be tuned. If it is too low, the MTP layers are under-utilized. If it is
too high, there may be too much tokens that cannot pass the validation and then be dropped.

Acceptable values: A positive integer.

*Default: `1`.*

### Argument `infer.language_model_only`

This parameter is only used for Qwen3.5 model family. If this parameter is true,
you can skip loading the vision encoder and only start the language model.

*Default: `False`.*

### Argument `infer.schedule_overlap`

Whether to overlap scheduling with tensor computation. We recommand to keep this feature
on whenever supported.

Acceptable values:
- "auto": Decide automatically.
- True: Enable overlapping.
- False: Disable overlapping.

*Default: `auto`.*

### Argument `infer.full_warmup`

If True, try to fully warmup each operator with many possible cases before launching the
service. It will take more time before the service is ready, but useful for reduce the
performance loss for the first several requests.

Acceptable values:
- "auto": Decide automatically by the following rules: Suppose you are benchmarking Chitu
  with a fixed context length, if will end up always running the decode stage with full
  batch size, then we set `infer.full_warmup=False`. Otherwise, we set `infer.full_warmup=True`.
- True: Fully warmup.
- False: Skip full warmup.

*Default: `auto`.*

### Argument `infer.process_group_timeout_seconds`

Override torch distributed process group timeout, in seconds.

Acceptable values:
- "auto": Decide automatically.
- null: Use the default from `torch.distributed`.
- An integer: the timeout in seconds.

*Default: `auto`.*

### Argument `infer.bind_process_to_cpu`

Whether and how to bind the currenct process to a CPU. If binding, it requires installing
chitu with `chitu[numa]` for extra dependency.

Acceptable values:
- "auto": Decide automatically.
- "none": Do not bind.
- "one_numa_per_rank": Bind each rank to a different NUMA node. This is helpful for CPU
  inference, where each rank is responsible for computing on a dedicated NUMA.
- "numa_near_device": Bind each rank to a NUMA node that is closest to the device this
  rank is responsible for. This is helpful for reducing CPU-GPU synchronizing latency.

*Default: `auto`.*

### Argument `infer.bind_thread_to_cpu`

How to bind threads to CPU cores for CPU inference. This field is ignored when CPUs are not
used for computing.

Acceptable values:
- "physical_core": Bind each thread to a physical core.
- "logical_core": Bind each thread to a logical core.

*Default: `physical_core`.*

### Argument `infer.enable_prefix_caching`

Whether to enable prefix caching
- True: enable
- False: not enable
-default: False

*Default: `False`.*

### Argument `infer.dp_prefix_caching_cache_threshold`

Minimum prefix-cache hit rate required to keep a task on the best prefix-cache DP rank.

*Default: `0.5`.*

### Argument `infer.dp_prefix_caching_balance_abs_threshold`

Absolute running-task spread required before DP preferred-rank selection ignores prefix locality.

*Default: `4`.*

### Argument `infer.dp_prefix_caching_balance_rel_threshold`

Relative running-task spread required before DP preferred-rank selection ignores prefix locality.

*Default: `1.5`.*

## `scheduler`

Configs for how chitu schedules multiple requests.

### Argument `scheduler.type`

Priority strategy. This field accepts an ordered comma separated list of stratigies. Requests
are first sorted by the leading strategy, and then by the next, and so on.

Acceptable values: An ordered comma separated list of scheduler types among:
- "fcfs": First come, first serve.
- "request_preset": Use priorities bound to API keys, set by `serve.api_keys`.
- "prefill_first": Priorities prefill first, then decode.
- "stride": Each task has a priority value P, and a score S (starts from 0), at scheduling point,
  update the scores: S += P * elapsed_time. Select the tasks with top scores and reset their
  scores back to 0.
- "deadline": Each task has a deadline time `DDL = request_arrival_time + prefix_tokens_len * alpha +
  max_output_tokens * beta`. Select the tasks with nearest DDL. Alpha and beta are arbitary value,
  defaults to 1ms.
- "prefix_align": Batch tasks with similar input lengths togather.

*Default: `"request_preset,prefill_first,fcfs"`.*

### `scheduler.pp_config`

Configs how to schedule for micro batches use for Pipeline Parallelism. Ignore when `pp_size == 1`.

#### Argument `scheduler.pp_config.pp_micro_batch_size_prefill`

Micro batching strategy for prefilling. This field has effect only when `pp_size > 1` and `cache_type`
is `paged`.

Acceptable values:
- "max": The maximum value of `prefill micro batch size` is limited to `max_reqs_per_dp / pp_size`.
- An integer: The maximum value of `prefill micro batch size` is limited to the value.
- "auto": Currently this means "max".

*Default: `auto`.*

#### Argument `scheduler.pp_config.pp_micro_batch_size_decode`

Micro batching strategy for decoding. This field has effect only when `pp_size > 1` and `cache_type`
is `paged`.

Acceptable values:
- "max": The maximum value of `decode micro batch size` is limited to `max_reqs_per_dp / pp_size`.
- An integer: The maximum value of `decode micro batch size` is limited to the number.
- "auto": Currently this means "max".

*Default: `auto`.*

## `multi_inst`

Configs for multi-instance depolyment including PD-disaggregation.

Note for per-instance config:
- `multi_inst.inst_id` and `multi_inst.router.is_router` should be set
   respectively for each instance.
-  In order for one instance to see other instances' config, all the other
   per-instance configs should be set via `multi_inst.inst_overrides` with
   their instance IDs.
-  All the other fields should be set to be the same during launch, but
   they will be overriden at run time according to
   `multi_inst.inst_overrides[multi_instinst_id]`.

### Argument `multi_inst.n_insts`

Number of instances.

*Default: `1`.*

### Argument `multi_inst.inst_id`

ID of the current instance.

This field should be set respectively for each instance.

It should be null for the router.

**This field can be omitted when launching from chitu.run (via `chitu.boot` module).**

*Default: `0`.*

### Argument `multi_inst.role`

Role of the current instance.

Acceptable values: "prefill_and_decode", "prefill", or "decode".

*Default: `"prefill_and_decode"`.*

### `multi_inst.pd_disaggregation`

Additional configs for PD disaggregation.

> TIPS: To enable verbose logging for PD disaggregation, set the following:
> `CHITU_LOGGING_LEVEL=chitu.distributed.pd_disaggregation:DEBUG;chitu.hooks:DEBUG;chitu.scheduler:DEBUG`

#### Argument `multi_inst.pd_disaggregation.kv_transfer_backend`

KV transfer backend

Acceptable values: mooncake, nccl

*Default: `"mooncake"`.*

#### Argument `multi_inst.pd_disaggregation.kv_transfer`

Additional configs for KV transfer

### Argument `multi_inst.inst_overrides`

Per-instance config overrides keyed by instance ID.

In order for one instance to see other instances' config, all per-instance
configs should be set in this override, except for torchrun arguments,
`multi_inst.inst_id`, and `multi_inst.router.is_router`.

Example:

```yaml
inst_overrides:
  0:
    infer:
      tp_size: 8
    multi_inst:
      role: "prefill"
      pd_disaggregation:
        prefill_scheduler:
          max_batch_size: 32
          max_total_tokens: 8192
          batching_strategy: "varlen"
  1:
    infer:
      dp_size: 8
      ep_size: 8
    multi_inst:
      role: "decode"
      pd_disaggregation:
        decode_scheduler:
          scheduling_strategy: "immediate"
```

Since passing a very-long dict may be a bad idea in CLI, you can pass this field
in either of the following two ways:

1. Passing a dict:
   ```
   'multi_inst.inst_overrides="{0: {infer: {tp_size: 8}, multi_inst: {role: "prefill", pd_disaggregation: {prefill_scheduler: {max_batch_size: 32}}}}}"'
   ```
2. First passing an empty dict, and then append to it:
   ```
   multi_inst.inst_overrides={} \
   +multi_inst.inst_overrides.0.infer.tp_size=8 \
   +multi_inst.inst_overrides.0.multi_inst.role="prefill" \
   +multi_inst.inst_overrides.0.multi_inst.pd_disaggregation.prefill_scheduler.max_batch_size=32
   ```

*Default: `null`.*

### `multi_inst.router`

Configs for router

#### Argument `multi_inst.router.is_router`

Set this to True if the current process is a router.

**This field can be omitted when launching from chitu.run (via `chitu.boot` module).**

*Default: `False`.*

#### Argument `multi_inst.router.launch_timeout`

When the router is ready, wait for this time untill the instances are ready.

*Default: `3600`.*

## `metrics`

Configs how chitu is monitored and logged.

### Argument `metrics.prometheus_listening_host`

Access prometheus server at this address

*Default: `0.0.0.0`.*

### Argument `metrics.prometheus_listening_port`

Access prometheus server at this port

*Default: `9090`.*

### Argument `metrics.prometheus_scrape_interval`

Prometheus server scrapes PrometheusMetricsCollectors every ${prometheus_scrape_interval} seconds.

*Default: `1`.*

### Argument `metrics.log_interval`

Print metrics every this seconds to terminal

*Default: `10`.*

### Argument `metrics.grafana_enabled`

Set to true to auto-start a Grafana server on rank 0

*Default: `false`.*

### Argument `metrics.grafana_host`

Grafana server bind address

*Default: `0.0.0.0`.*

### Argument `metrics.grafana_port`

Grafana HTTP port

*Default: `9095`.*

## `debug`

Debugging options. These options have negative effect on performance and correctness, so please
don't set them in production.

### Argument `debug.skip_model_load`

Skip model loading and run on uninitialized weights. You will NOT get correct output with this
enabled. This is useful for quick debugging.

Acceptable values:
- False: Normal behavior.
- True: Skip model loading.

*Default: `False`.*

### Argument `debug.force_moe_balance`

Force MoE gate to chose some balanced experts. You will NOT get correct output with this enabled.
This is useful for analyze the performance inpact of MoE load inbalance, by comparing performance
with this option on and off.

Acceptable values:
- False: Normal behavior.
- True: Force balance.

*Default: `False`.*

### Argument `debug.save_trace_dir`

Save trace to a directory. The trace can be replayed with `benchmarks/benchmark_serving.py` with
additional `--dataset chitu-trace --dataset-path <path/to/trace>` arguments.

Acceptable values:
- null: No trace will be saved.
- /path/to/directory: Save trace to this directory.

*Default: `null`.*

### Argument `debug.disable_inter_op_auto_tune`

If true, skip heavy inter-op auto-tuning. The service will start quickly but the performance
may be worse.

Acceptable values:
- False: Normal behavior.
- True: Skip inter-op auto-tuning.

*Default: `False`.*

## Argument `float_16bit_variant`

The data type for 16-bit floating point data type. This field is orthogonal to quantization.

Acceptable values:
- "bfloat16": Wider range, less precision. There may be accuracy loss for small (<= ~7B) models.
- "float16": Narrower range, higher precision. But some models will result in NaN.

*Default: `bfloat16`.*

## Argument `use_float32_rotary`

Data type for RoPE (rotary positional encoding). Setting to float32 may be helpful if the
context length is very long.

Acceptable values:
- True: Use float32 for RoPE.
- False: Use the same dtype as `float_16bit_variant`.

*Default: `False`.*

## Argument `keep_dtype_in_checkpoint`

What to do if the data type mismatches between the model definition (the code) and the checkpoint
(the model file on disk). Whatever is this field, you will receive a warning when the data type
mismatches.

Acceptable values:
- True: Use the data type in the checkpoint.
- False: Use the data type in the model definition.

*Default: `False`.*

## Argument `skip_preprocess`

When using `script/preprocess_and_save.py` to preprocess a model's state dict, the preprocessed
file can be loaded via setting this field to True. This is useful if the running node has constrained
file system size. See `docs/en/DEVELOPMENT.md` for details.

*Default: `False`.*

## Argument `gpu_preprocess`

Preprocess state dict using GPU to accelerate model load time, but use a bit more GPU memory
during loading. Only takes effect when using layerwise loading.

*Default: `True`.*

## Argument `disable_layerwise_load`

Disable layerwise checkpoint loading. When True, falls back to loading the
full checkpoint at once instead of streaming layer-by-layer. Useful on
platforms where layerwise loading causes memory fragmentation issues (e.g. Ascend NPU).

*Default: `False`.*

## Argument `model_load_per_layer_timeout_s`

Timeout for each process to finish one model layer during layerwise loading, in seconds.

This is an explicit timeout effective for layerwise loading. For non-layerwise loading, implicit
timeout may or may not happen in the communication immedately after model loading.

*Default: `60`.*
