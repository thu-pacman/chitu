# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf
import sys
from logging import getLogger

logger = getLogger(__name__)


class ServeConfigRules(Callback):
    def __init__(self) -> None:
        super().__init__()

    def _exit_with_error(self, message):
        """Fatal error, exit method"""
        logger.error(f"Config Error: {message}")
        sys.exit(1)

    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        port = config.serve.port
        if not (1024 <= port <= 65535):
            self._exit_with_error(f"Port must be between 1024 and 65535, got {port}")

        num_blocks = config.infer.num_blocks
        if num_blocks < 0 and num_blocks != -1:
            self._exit_with_error(
                f"num_blocks must be positive or -1 (got {num_blocks})"
            )

        attn_type = config.infer.attn_type
        if attn_type == "npu":
            try:
                import torch_npu
            except ImportError:
                self._exit_with_error(
                    f"torch-npu required for attn_type=npu (got {attn_type})"
                )
        if attn_type not in {
            "auto",
            "flash_attn",
            "flash_mla",
            "flash_infer",
            "dllm",
            "hunyuan_attn",
            "triton",
            "npu",
            "hopper_mixed",
            "ref",
        }:
            self._exit_with_error(
                f"attn_type must be one of [auto, flash_attn, flash_mla, flash_infer, hunyuan_attn, triton, npu, hopper_mixed, ref], got {attn_type}"
            )

        model_name = config.models.name
        model_type = config.models.type
        if attn_type == "flash_infer":
            if config.models.n_heads % config.models.n_kv_heads != 0:
                self._exit_with_error(
                    f"model {model_name} is not compatible with flash_infer: "
                    f"n_heads ({config.models.n_heads}) must be divisible by "
                    f"n_kv_heads ({config.models.n_kv_heads})"
                )
        elif attn_type == "flash_mla":
            if model_type not in [
                "deepseek-v3",
                "deepseek-v4",
                "kimi-k2-5",
                "glm-5-2",
                "glm-5-next",
            ]:
                self._exit_with_error(
                    f"model {model_name} is not compatible with flash_mla"
                )

        if model_type == "hf-gpt-oss" and attn_type != "ref":
            self._exit_with_error(f"model {model_name} is only compatible with ref")

        tokenizer_type = config.models.tokenizer_type
        if tokenizer_type not in {"hf", "tiktoken"}:
            self._exit_with_error(
                f"tokenizer_type must be one of [hf, tiktoken], got {tokenizer_type}"
            )

        op_impl = config.infer.op_impl
        if op_impl not in {"torch", "muxi_custom_kernel", "cpu"}:
            self._exit_with_error(
                f"op_impl must be one of [torch, muxi_custom_kernel, cpu], got {op_impl}"
            )

        mla_absorb = config.infer.get("mla_absorb", "auto")
        if mla_absorb not in {
            "auto",
            "none",
            "absorb-without-precomp",
            "absorb-kv-only",
            "absorb",
        }:
            self._exit_with_error(
                "mla_absorb must be one of "
                "[auto, none, absorb-without-precomp, absorb-kv-only, absorb], "
                f"got {mla_absorb}"
            )

        bind_process_to_cpu = config.infer.bind_process_to_cpu
        if bind_process_to_cpu not in {
            "auto",
            "none",
            "one_numa_per_rank",
            "numa_near_device",
        }:
            self._exit_with_error(
                f"bind_process_to_cpu must be one of [auto, none, one_numa_per_rank numa_near_device], got {bind_process_to_cpu}"
            )

        bind_thread_to_cpu = config.infer.bind_thread_to_cpu
        if bind_thread_to_cpu not in {"physical_core", "logical_core"}:
            self._exit_with_error(
                f"bind_thread_to_cpu must be one of [physical_core, logical_core], got {bind_thread_to_cpu}"
            )

        multi_inst = config.multi_inst
        if multi_inst.router.is_router and multi_inst.n_insts <= 1:
            self._exit_with_error(
                f"multi_inst.n_insts must be greater than 1 when multi_inst.router.is_router is true, got {multi_inst.n_insts}"
            )

        self._check_linear_checkpoint_interval(config)
        self._check_linear_attention_impl(config)
        self._check_pcp_with_linear_attention(config)

    def _check_linear_attention_impl(self, config: DictConfig) -> None:
        """linear attention 的算子实现必须能写 checkpoint。

        linear 的 prefix caching 靠算子按 chunk 写出的中间 state，目前只有 torch 实现支持
        （见 chitu/ops/linear_attn.py、chitu/ops/causal_conv.py 的 register_auto）。显式指定
        fla 会绕过自动选择，在第一次 prefill 时直接 assert 失败，这里提前报错。
        """
        linear_attn_model_types = {"hf-qwen3-next", "hf-qwen3-5"}
        if config.models.type not in linear_attn_model_types:
            return
        if not bool(config.infer.enable_prefix_caching):
            return
        # decode 实例整体不做 linear 的 prefix caching（见 builders._linear_checkpoint_interval），
        # 没有 checkpoint 要写，什么实现都能用
        if getattr(config.multi_inst, "role", None) == "decode":
            return
        impl = config.models.get("linear_attention_impl", "auto")
        if impl == "fla":
            self._exit_with_error(
                f"linear_attention_impl={impl} cannot write the intermediate linear-attention "
                'states that prefix caching needs; use "auto" (which falls back to the '
                'torch impl when checkpoints are requested) or "torch"'
            )

    def _check_linear_checkpoint_interval(self, config: DictConfig) -> None:
        """Validate infer.linear_checkpoint_interval against prefix caching.

        The linear-attention (GDN) recurrent state can only be reused from a
        checkpoint every C tokens, so prefix caching on such a model needs C.
        Leaving it null is allowed: the builders derive C from
        the main KV cache block size (_default_linear_checkpoint_interval), so this
        only validates explicitly given values.
        """
        # Model types whose KV cache spec includes a "linear" cache, i.e. the ones
        # with linear-attention layers. Keep in sync with the kv_cache spec registry
        # (chitu/kv_cache/providers/qwen.py).
        linear_attn_model_types = {"hf-qwen3-next", "hf-qwen3-5"}

        interval = config.infer.get("linear_checkpoint_interval", None)
        model_type = config.models.type

        if interval is None:
            return

        enable_prefix_caching = bool(config.infer.enable_prefix_caching)

        if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
            self._exit_with_error(
                f"linear_checkpoint_interval must be a positive integer or null, got {interval}"
            )
        if not enable_prefix_caching:
            self._exit_with_error(
                f"linear_checkpoint_interval={interval} has no effect unless "
                "enable_prefix_caching is true"
            )
        if model_type not in linear_attn_model_types:
            self._exit_with_error(
                f"linear_checkpoint_interval is only supported by models with "
                f"linear-attention layers ({sorted(linear_attn_model_types)}), "
                f"but model {config.models.name} has type {model_type}"
            )
        if interval > config.infer.max_seq_len:
            self._exit_with_error(
                f"linear_checkpoint_interval={interval} must not exceed "
                f"infer.max_seq_len ({config.infer.max_seq_len}): a checkpoint "
                "would never be written"
            )
        prefill_chunk_size = config.infer.prefill_chunk_size
        if isinstance(prefill_chunk_size, int) and interval > prefill_chunk_size:
            # The prefill scheduler never gives a task a chunk shorter than C
            # unless the whole task fits in the remaining budget of that step,
            # so a task longer than the chunk size would never be scheduled.
            self._exit_with_error(
                f"linear_checkpoint_interval={interval} must not exceed "
                f"infer.prefill_chunk_size ({prefill_chunk_size})"
            )

    def _check_pcp_with_linear_attention(self, config: DictConfig) -> None:
        """带线性注意力层的模型对 pcp（prefill context parallel）的支持程度。

        线性注意力的短卷积和递归 state 要求 token 相邻，而 cp 会把 prefill 的 token 按
        ``i % pcp_size`` 交错切给各 rank（见 chitu/cp_utils.py 的 split_prefill /
        prepare_local_lengths）。所以能不能开 cp，取决于各 rank 拿到的是不是「全局位置」
        的输入 —— 是全局位置，state / checkpoint 就与 cp=1 时逐字节一致。

        - ``glm-5-next``（chitu/models/model_glm5_next.py）：已支持 cp prefill。层里把投影
        后的 qkv/beta/f_a/g_a all-gather 成全局序列，在全局序列上跑完短卷积和递归，再把本
        rank 的行切回来（索引器同理），所以每个 rank 算出的 state 都是全局位置上的量；写
        进 page 的内容也就与 cp=1 时相同（cp rank 之间只是重复计算）。而 prefix caching 只
        按「序列绝对位置」决定 state 落在哪一页（``_ckpt_page_id`` / ``_upd_ckpt_write_pages``
        用绝对 position ids，chunk 起点由调度器对齐到 C 的网格），与 cp 怎么切无关，所以
        cp 和 prefix caching 可以同时开。注意 ``infer.pcp_size > 1`` 还有并行度上的限制
        （``dp_size`` 必须为 1，见 chitu/boot/arg_utils.py），那是另一回事。
        - ``hf-qwen3-next`` / ``hf-qwen3-5``：模型里还没有 cp 路径，层拿到本 rank 的
        token 后仍把全局的 ``seq_len_delta`` 当 cu_seqlens 传给算子（见
        Qwen3NextGatedDeltaNet.forward），算子的 grid/索引按全局长度算会直接越界
        （CUDA illegal memory access）；即使换成本地长度，交错切分下结果也是错的。
        这两个模型一律拒绝 cp。
        """
        # 有线性注意力层的模型类型，见 chitu/models/model_hf_qwen3_next.py、
        # chitu/models/model_hf_qwen3_5.py、chitu/models/model_glm5_next.py
        linear_attn_model_types = {"hf-qwen3-next", "hf-qwen3-5", "glm-5-next"}
        # 已经有 cp prefill 实现的模型类型（层内 all-gather 出全局序列），其余一律不支持 cp
        cp_prefill_model_types = {"glm-5-next"}
        model_type = config.models.type
        if model_type not in linear_attn_model_types:
            return
        if model_type in cp_prefill_model_types:
            return

        def _reject(pcp_size, where: str) -> None:
            self._exit_with_error(
                f"model {config.models.name} (type {model_type}) has linear-attention "
                f"layers, which do not support infer.pcp_size > 1 (got "
                f"infer.pcp_size={pcp_size}{where}). Use infer.pcp_size=1 "
                "for this model."
            )

        # 单实例
        if int(config.infer.pcp_size) > 1:
            _reject(config.infer.pcp_size, "")

        # 多实例（PD / 混部）
        inst_overrides = getattr(config.multi_inst, "inst_overrides", None) or {}
        for inst_id, override in sorted(
            inst_overrides.items(), key=lambda kv: str(kv[0])
        ):
            merged = OmegaConf.merge(config, override)
            pcp_size = int(merged.infer.pcp_size)
            if pcp_size > 1:
                _reject(pcp_size, f" in instance {inst_id}")
