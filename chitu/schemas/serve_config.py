from omegaconf import MISSING
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ServeAddrConfig:
    host: str = MISSING
    port: int = MISSING


@dataclass
class InferConfig:
    tp_size: int = MISSING
    pp_size: int = MISSING
    do_load: bool = MISSING
    seed: float = MISSING
    max_seq_len: int = MISSING
    cache_type: str = MISSING
    attn_type: str = MISSING
    op_impl: str = MISSING
    mla_absorb: Optional[str] = MISSING
    raise_lower_bit_float_to: str = MISSING
    soft_fp8: bool = MISSING  # Legacy parameter. To be removed in the future.
    fuse_shared_experts: bool = MISSING
    max_reqs: int = MISSING
    pp_layer_partition: Optional[list] = MISSING
    use_cuda_graph: bool = MISSING
    num_blocks: int = MISSING
    bind_process_to_cpu: Optional[str] = MISSING
    bind_thread_to_cpu: str = MISSING
    gpu_memory_utilization: float = MISSING


@dataclass
class RequestConfig:
    prompt_tokens_len: int = MISSING
    max_new_tokens: int = MISSING


@dataclass
class SchedulerConfig:
    @dataclass
    class SchedulerCommonConfig:
        num_tasks: Optional[int] = MISSING
        enable_hybrid: bool = MISSING

    @dataclass
    class PrefillFirstConfig:
        @dataclass
        class PpConfig:
            prefill_num_tasks_divided_by_pp: bool = MISSING
            prefill_num_tasks: Optional[int] = MISSING
            enforce_decoder_num_tasks_max: bool = MISSING
            decoder_num_tasks: Optional[int] = MISSING

        num_tasks: Optional[int] = MISSING
        enable_hybrid: bool = MISSING
        pp_config: PpConfig = MISSING

    type: str = MISSING
    fcfs: SchedulerCommonConfig = MISSING
    prefill_first: PrefillFirstConfig = MISSING
    stride: SchedulerCommonConfig = MISSING
    deadline: SchedulerCommonConfig = MISSING
    prefix_align: SchedulerCommonConfig = MISSING
    balance: SchedulerCommonConfig = MISSING


@dataclass
class ServeConfig:
    serve: ServeAddrConfig = field(default_factory=ServeAddrConfig)
    models: Any = MISSING
    infer: InferConfig = field(default_factory=InferConfig)
    request: RequestConfig = field(default_factory=RequestConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    quant: Optional[str] = MISSING
    dtype: Optional[str] = MISSING  # Legacy parameter. To be removed in the future.
    float_16bit_variant: str = MISSING
    keep_dtype_in_checkpoint: bool = MISSING
    skip_preprocess: bool = MISSING
