# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Union
from omegaconf import MISSING, OmegaConf

######################################################################################
# The following are legacy configs. They might be removed at any time in the future.


@dataclass
class InferConfigLegacy:
    do_load: bool = MISSING
    soft_fp8: bool = MISSING
    max_reqs: Optional[int] = MISSING


@dataclass
class PpConfigLegacy:
    prefill_num_tasks_divided_by_pp: bool = MISSING
    prefill_num_tasks: Optional[int] = MISSING
    enforce_decode_num_tasks_max: bool = MISSING
    decode_num_tasks: Optional[int] = MISSING


@dataclass
class ServeConfigLegacy:
    dtype: Optional[str] = MISSING


######################################################################################
# The following are active configs in use.


@dataclass
class ApiKey:
    key: str = MISSING
    priority: int = 1


@dataclass
class ServeAddrConfig:
    host: str = MISSING
    port: int = MISSING
    api_keys: list[ApiKey] = field(default_factory=list)
    validate_api_key: bool = MISSING
    # Optional: map external model names (e.g. Anthropic) to the currently loaded internal model name.
    # Example:
    #   model_aliases:
    #     claude-3-5-sonnet-latest: DeepSeek-V3.1
    model_aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class InferConfig(InferConfigLegacy):
    pcp_size: int = MISSING
    tp_size: int = MISSING
    pp_size: int = MISSING
    dp_size: int = MISSING
    ep_size: int = MISSING
    etp_size: Optional[int] = MISSING
    seed: float = MISSING
    max_seq_len: int = MISSING
    cache_type: str = MISSING
    indexer_type: str = MISSING
    attn_type: str = MISSING
    op_impl: str = MISSING
    mla_absorb: Optional[str] = MISSING
    raise_lower_bit_float_to: str = MISSING
    fuse_shared_experts: bool = MISSING
    max_batch_size: int = MISSING
    max_concurrent_requests: Optional[int] = MISSING
    device_ids: Optional[list[int]] = MISSING
    pp_layer_partition: Optional[list[int]] = MISSING
    use_cuda_graph: bool | str = MISSING
    minimax_sparse_decode_backend: str = MISSING
    minimax_sparse_prefill_backend: str = MISSING
    npu_fusion_fp4: bool = MISSING
    num_blocks: int = MISSING
    max_multimodal_blocks: int = MISSING
    bind_process_to_cpu: str = MISSING
    bind_thread_to_cpu: str = MISSING
    memory_utilization: float = MISSING
    prefill_chunk_size: Union[int, str, None] = MISSING
    schedule_overlap: bool | str = MISSING
    full_warmup: bool | str = MISSING
    process_group_timeout_seconds: Optional[int | str] = MISSING
    embed_tokens_lm_head_tp_size: str = MISSING
    experts_stats_path: Optional[str] = MISSING
    num_experts_slots: Optional[int] = MISSING
    moe_lb_trigger: int = MISSING
    moe_lb_threshold: float = MISSING
    dllm_block_length: int = MISSING  # block length for dLLM decode
    enable_prefix_caching: bool = MISSING
    dp_prefix_caching_hit_rate_weight: float = MISSING
    dp_prefix_caching_running_penalty_weight: float = MISSING

    @dataclass
    class MoEConfig:
        prefill_memory_tolerance: float = MISSING
        prefill_token_dispatcher: str = MISSING
        decode_token_dispatcher: str = MISSING

    moe: MoEConfig = MISSING
    mtp_size: int = MISSING
    language_model_only: bool = MISSING


@dataclass
class RequestConfig:
    prompt_tokens_len: int = MISSING
    max_new_tokens: int = MISSING
    frequency_penalty: float = MISSING


@dataclass
class PpConfig(PpConfigLegacy):
    pp_micro_batch_size_prefill: str = MISSING
    pp_micro_batch_size_decode: str = MISSING


@dataclass
class SchedulerConfig:
    pp_config: PpConfig = MISSING
    type: str = MISSING


@dataclass
class KvTransferConfig:
    buffer_size: int = MISSING
    transfer_timeout: float = MISSING
    max_concurrent_transfers: int = MISSING
    # Settings for how long Decode waits for the Success signal from Prefill
    # (see kv_manager.recv_kv_cache_and_insert).
    decode_wait_timeout_s: float = MISSING
    decode_resend_interval_s: float = MISSING
    decode_poll_interval_s: float = MISSING
    # Decode preallocation settings. These separate preallocation from runtime work.
    decode_prealloc_max_pending: Optional[int] = MISSING
    decode_prealloc_poll_interval_s: float = MISSING
    # Decode preallocation budget. Limits the number of tokens that can be
    # preallocated at the same time. A value <= 0 disables this limit.
    decode_prealloc_token_budget: Optional[int] = MISSING
    decode_prealloc_reserved_tokens: int = MISSING
    # Decode runtime concurrency limit, per DP rank. This separates the block
    # limit from the runtime concurrency limit. A value <= 0 means no limit.
    decode_max_running_tasks_per_dp: Optional[int] = MISSING
    # Log throttling interval for Decode prepare backpressure, in seconds.
    prepare_backpressure_log_interval_s: float = MISSING


@dataclass
class PDTestConfig:
    """PD smoke-test configuration."""

    # 0=off, 1=basic test, 2=test with shared system prompt (prefix-cache-friendly)
    enable: int = MISSING
    req_num: int = MISSING  # number of test requests
    req_timeout: float = MISSING  # per-request timeout (seconds)
    output_len: int = MISSING  # max_new_tokens for each test request


@dataclass
class PrefillSchedulerConfig:
    """Prefill Scheduler configuration

    Each prefill scheduler binds to a random port and registers it in the
    coordinator under the role ``prefill_instance_<id>``; the router discovers
    it via the coordinator. See ``chitu/distributed/coordinator.py``.
    """

    max_batch_size: int = MISSING
    max_total_tokens: int = MISSING
    batching_strategy: str = MISSING  # varlen, fixed


@dataclass
class DecodeSchedulerConfig:
    """Decode Scheduler configuration

    Each decode scheduler binds to a random port and registers it in the
    coordinator under the role ``decode_instance_<id>``; the router discovers
    it via the coordinator. See ``chitu/distributed/coordinator.py``.
    """

    scheduling_strategy: str = MISSING  # immediate, batched


@dataclass
class PDDisaggregationConfig:
    """PD disaggregation configuration"""

    prefill_scheduler: Optional[PrefillSchedulerConfig] = MISSING
    decode_scheduler: Optional[DecodeSchedulerConfig] = MISSING
    kv_transfer_backend: str = MISSING  # kv transfer backend: mooncake, nccl
    kv_transfer: KvTransferConfig = field(default_factory=KvTransferConfig)


@dataclass
class RouterConfig:
    is_router: bool = MISSING
    max_inflight_per_instance: int = MISSING
    routing_algorithm: str = MISSING
    routing_algorithm_for_decode: str = MISSING
    router_cache_miss_fallback_algorithm: str = MISSING
    router_hit_weight: float = MISSING
    router_load_penalty_weight: float = MISSING
    router_evict_buffer_size: int = MISSING
    router_local_reservation_timeout_s: float = MISSING
    launch_timeout: float = MISSING


@dataclass
class CoordinatorConfig:
    host: Optional[str] = MISSING
    port: Optional[int] = MISSING


@dataclass
class MultiInstConfig:
    n_insts: int = MISSING
    inst_id: Optional[int] = MISSING
    role: str = MISSING
    pd_disaggregation: PDDisaggregationConfig = field(
        default_factory=PDDisaggregationConfig
    )

    # NOTE 1: Although this is a dict, please keep typing of this field as Any
    #         so Hydra can override arbitrary instance-ID keys without requiring
    #         `+`.
    # NOTE 2: Keys are cast to int at runtime, so passing either "1" or 1 from
    #         YAML or CLI is fine.
    inst_overrides: Any = field(default_factory=dict)

    router: RouterConfig = MISSING


@dataclass
class MetricsConfig:
    """Metrics collection configuration"""

    prometheus_listening_host: str = MISSING
    prometheus_listening_port: int = MISSING
    prometheus_config_file: str = MISSING
    prometheus_data_dir: str = MISSING
    prometheus_scrape_interval: int = MISSING
    log_interval: int = MISSING
    grafana_enabled: bool = MISSING
    grafana_host: str = MISSING
    grafana_port: int = MISSING


@dataclass
class DebugConfig:
    skip_model_load: bool = MISSING
    force_moe_balance: bool = MISSING
    save_trace_dir: Optional[str] = MISSING
    disable_inter_op_auto_tune: bool = MISSING


class StaticConfig:
    def __init__(self, config_obj):
        if hasattr(config_obj, "__dataclass_fields__"):
            self._data = asdict(config_obj)
        elif isinstance(config_obj, dict):
            self._data = config_obj
        else:
            self._data = OmegaConf.to_container(config_obj, resolve=True)
        self._convert_nested_structures()

    def _convert_nested_structures(self):
        for k, v in self._data.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, dict):
                setattr(self, k, StaticConfig(v))
            elif isinstance(v, list):
                setattr(self, k, self._convert_list(v))
            else:
                setattr(self, k, v)

    def _convert_list(self, lst):
        result = []
        for item in lst:
            if isinstance(item, dict):
                result.append(StaticConfig(item))
            elif isinstance(item, list):
                result.append(self._convert_list(item))
            else:
                result.append(item)
        return result

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                static_value = StaticConfig(value)
                setattr(self, name, static_value)
                return static_value
            return value

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return [getattr(self, k) for k in self._data.keys()]

    def items(self):
        return [(k, getattr(self, k)) for k in self._data.keys()]

    def __repr__(self):
        return f"StaticConfig({self._data!r})"


@dataclass
class ServeConfig(ServeConfigLegacy):
    boot: Any = MISSING
    serve: ServeAddrConfig = field(default_factory=ServeAddrConfig)
    models: Any = MISSING
    benchmark: Any = MISSING
    infer: InferConfig = field(default_factory=InferConfig)
    request: RequestConfig = field(default_factory=RequestConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    multi_inst: MultiInstConfig = field(default_factory=MultiInstConfig)
    pd_test: PDTestConfig = field(default_factory=PDTestConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    quant: Optional[str] = MISSING
    gpu_preprocess: bool = MISSING
    disable_layerwise_load: bool = MISSING
    model_load_per_layer_timeout_s: float = MISSING
    float_16bit_variant: str = MISSING
    use_float32_rotary: bool = MISSING
    keep_dtype_in_checkpoint: bool = MISSING
    skip_preprocess: bool = MISSING

    def to_object(self):
        return StaticConfig(self)
