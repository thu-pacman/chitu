# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from omegaconf import MISSING

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
    tp_size: int = MISSING
    pp_size: int = MISSING
    dp_size: int = MISSING
    ep_size: int = MISSING
    seed: float = MISSING
    max_seq_len: int = MISSING
    cache_type: str = MISSING
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
    npu_fusion_fp4: bool = MISSING
    num_blocks: int = MISSING
    max_multimodal_blocks: int = -1
    bind_process_to_cpu: str = MISSING
    bind_thread_to_cpu: str = MISSING
    memory_utilization: float = MISSING
    prefill_chunk_size: Union[int, str, None] = MISSING
    schedule_overlap: bool | str = MISSING
    full_warmup: bool | str = MISSING
    embed_tokens_lm_head_tp_size: str = MISSING
    experts_stats_path: Optional[str] = None
    num_experts_slots: Optional[int] = None
    moe_lb_trigger: int = -1
    moe_lb_threshold: float = 3.0
    enable_prefix_caching: bool = MISSING
    dp_prefix_caching_idle_rate_weight: float = 0.01
    dp_prefix_caching_running_penalty_weight: float = 0.01

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
class DpAddressesConfig:
    host: str = MISSING
    port: int = MISSING


@dataclass
class KvTransferConfig:
    buffer_size: int = 2048
    transfer_timeout: float = 30.0
    max_concurrent_transfers: int = 8
    # Decode 侧等待 Prefill 回写 Success 信号的参数（见 kv_manager.recv_kv_cache_and_insert）
    decode_wait_timeout_s: float = 20.0
    decode_resend_interval_s: float = 0.5
    decode_poll_interval_s: float = 0.05
    # Decode 预分配相关参数：用于解耦预分配与运行时
    decode_prealloc_max_pending: Optional[int] = None
    decode_prealloc_poll_interval_s: float = 0.01
    # Decode 预分配预算：控制 prealloc 并发的 token 数量
    decode_prealloc_token_budget: Optional[int] = None
    decode_prealloc_reserved_tokens: int = 0
    # Decode 运行并发上限（per DP rank）。用于将 block 上限与运行并发解耦
    decode_max_running_tasks_per_dp: Optional[int] = None
    # Decode prepare backpressure 日志节流间隔/s
    prepare_backpressure_log_interval_s: float = 5.0


@dataclass
class PDDisaggregationConfig:
    """PD disaggregation configuration"""

    enabled: bool = False
    coordination_port: int = 29800  # P-D coordination port
    metadata_sync_port: int = 29801  # metadata sync port
    kv_transfer_backend: str = "mooncake"  # kv transfer backend: mooncake, nccl
    ib_device: Optional[str] = "mlx5_0"  # IB device name
    bootstrap_port: int = 8080  # Bootstrap server port
    # High-frequency PD logs (PD_QUEUE/PD_STATS/PD_TRACE)
    log_verbose: bool = False
    kv_transfer: KvTransferConfig = field(default_factory=KvTransferConfig)


@dataclass
class PrefillSchedulerConfig:
    """Prefill Scheduler configuration"""

    host: str = MISSING
    port: int = MISSING
    max_batch_size: int = MISSING
    max_total_tokens: int = MISSING
    batching_strategy: str = MISSING  # varlen, fixed
    # kv_config: KvTransferConfig = MISSING


@dataclass
class DecodeSchedulerConfig:
    """Decode Scheduler configuration"""

    host: str = MISSING
    port: int = MISSING
    scheduling_strategy: str = MISSING  # immediate, batched
    # kv_config: KvTransferConfig = MISSING


@dataclass
class RouterConfig:
    is_router: bool = MISSING
    host: str = MISSING
    port: int = MISSING
    stats_port: int = MISSING
    token_port: int = MISSING
    load_balancer_algorithm: str = MISSING
    dp_addresses: list[DpAddressesConfig] = MISSING
    # PD disaggregation configuration
    pd_disaggregation: PDDisaggregationConfig = field(
        default_factory=PDDisaggregationConfig
    )
    prefill_schedulers: list[PrefillSchedulerConfig] = field(default_factory=list)
    decode_schedulers: list[DecodeSchedulerConfig] = field(default_factory=list)


@dataclass
class DpConfig:
    enabled: bool = MISSING
    scheduler_base_host: str = MISSING
    scheduler_base_port: int = MISSING
    dp_size: int = MISSING
    dp_id: int = MISSING
    tp_size: int = MISSING
    pp_size: int = MISSING
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


class StaticConfig:
    def __init__(self, config_obj):
        if hasattr(config_obj, "__dataclass_fields__"):
            from dataclasses import asdict

            self._data = asdict(config_obj)
        elif hasattr(config_obj, "__dict__"):
            self._data = config_obj.__dict__
        elif isinstance(config_obj, dict):
            self._data = config_obj
        else:
            try:
                from omegaconf import OmegaConf

                self._data = OmegaConf.to_container(config_obj, resolve=True)
            except Exception:
                self._data = {}
        self._convert_nested_structures()

    def _convert_nested_structures(self):
        for k, v in self._data.items():
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
    serve: ServeAddrConfig = field(default_factory=ServeAddrConfig)
    models: Any = MISSING
    benchmark: Any = MISSING
    infer: InferConfig = field(default_factory=InferConfig)
    request: RequestConfig = field(default_factory=RequestConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    dp_config: DpConfig = field(default_factory=DpConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    quant: Optional[str] = MISSING
    float_16bit_variant: str = MISSING
    use_float32_rotary: bool = MISSING
    keep_dtype_in_checkpoint: bool = MISSING
    skip_preprocess: bool = MISSING

    def to_object(self):
        return StaticConfig(self)
