# Chitu 指标

> 本文档由 `script/generate_metrics_docs.py` 生成。请勿手动编辑；如需修改，请更新 `chitu/metrics/definitions.py` 并重新运行该脚本。

## 概览

Chitu 暴露 Prometheus 指标，用于调试吞吐、请求延迟、KV cache 使用率、加速器显存、PD 分离和错误情况。原始指标和派生指标定义在带类型的 Python registry 中，本文档由该 registry 生成。

## 命名和类型说明

Python Prometheus client 创建的 Counter 查询名会使用 `_total` 后缀。例如，`chitu_total_generated_tokens` 的查询名为 `chitu_total_generated_tokens_total`，而 `chitu_completed_requests_total` 因为已经带有该后缀，查询名保持不变。Histogram 会暴露 `_bucket`、`_count` 和 `_sum` 序列。Ratio 指标的取值范围为 0 到 1。

## 原始指标

### KV cache

#### `chitu_kv_cache_usage_ratio`

KV cache 使用率（used_blocks / total_blocks）。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_kv_cache_usage_ratio` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | ratio |
| 范围 | `per_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_used_blocks`

KV cache 已使用 block 数。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_used_blocks` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | 无 |
| 范围 | `per_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_total_blocks`

KV cache 总 block 数。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_total_blocks` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | 无 |
| 范围 | `per_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_prealloc_blocks`

PD 分离预分配 block 数量。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_prealloc_blocks` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | 无 |
| 范围 | `per_instance` |
| Stdout | 是 |
| Grafana | 否 |

#### `chitu_total_hit_tokens_total`

Prefix cache 命中的 prompt token 总数。

| 属性 | 值 |
|---|---|
| 类型 | `counter` |
| Prometheus 查询名 | `chitu_total_hit_tokens_total` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | 无 |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_total_task_evictions_total`

由于 KV cache 不足而被驱逐的任务总数。

| 属性 | 值 |
|---|---|
| 类型 | `counter` |
| Prometheus 查询名 | `chitu_total_task_evictions_total` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | 无 |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |


### PD 分离

#### `chitu_pd_stage_duration_seconds`

PD 分离各阶段请求延迟（秒）。

| 属性 | 值 |
|---|---|
| 类型 | `histogram` |
| Prometheus 查询名 | `chitu_pd_stage_duration_seconds` |
| 标签 | `role`, `stage` |
| 单位 | seconds |
| 范围 | `per_role` |
| Stdout | 否 |
| Grafana | 是 |
| Prometheus 还会暴露 | `chitu_pd_stage_duration_seconds_bucket`, `chitu_pd_stage_duration_seconds_count`, `chitu_pd_stage_duration_seconds_sum` |

#### `chitu_kv_transfer_duration_seconds`

KV cache 传输耗时（秒）。

| 属性 | 值 |
|---|---|
| 类型 | `histogram` |
| Prometheus 查询名 | `chitu_kv_transfer_duration_seconds` |
| 标签 | `instance_id`, `rank` |
| 单位 | seconds |
| 范围 | `per_rank` |
| Stdout | 否 |
| Grafana | 是 |
| Prometheus 还会暴露 | `chitu_kv_transfer_duration_seconds_bucket`, `chitu_kv_transfer_duration_seconds_count`, `chitu_kv_transfer_duration_seconds_sum` |

#### `chitu_kv_transfer_size_bytes`

每个请求的 KV cache 传输大小（字节）。

| 属性 | 值 |
|---|---|
| 类型 | `histogram` |
| Prometheus 查询名 | `chitu_kv_transfer_size_bytes` |
| 标签 | `instance_id`, `rank` |
| 单位 | bytes |
| 范围 | `per_rank` |
| Stdout | 否 |
| Grafana | 否 |
| Prometheus 还会暴露 | `chitu_kv_transfer_size_bytes_bucket`, `chitu_kv_transfer_size_bytes_count`, `chitu_kv_transfer_size_bytes_sum` |

#### `chitu_pd_queue_size`

PD 分离各阶段当前队列长度。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_pd_queue_size` |
| 标签 | `role`, `queue_name` |
| 单位 | 无 |
| 范围 | `per_role` |
| Stdout | 否 |
| Grafana | 是 |


### 吞吐

#### `chitu_total_generated_tokens_total`

Executor 生成 token 总数。

| 属性 | 值 |
|---|---|
| 类型 | `counter` |
| Prometheus 查询名 | `chitu_total_generated_tokens_total` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | 无 |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_total_prompt_tokens_total`

Executor 处理 prompt token 总数。

| 属性 | 值 |
|---|---|
| 类型 | `counter` |
| Prometheus 查询名 | `chitu_total_prompt_tokens_total` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | 无 |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_mtp_proposed_tokens_total`

MTP 提议 token 总数（每个任务每个 decode step 为 mtp_size-1）。

| 属性 | 值 |
|---|---|
| 类型 | `counter` |
| Prometheus 查询名 | `chitu_mtp_proposed_tokens_total` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | 无 |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_mtp_accepted_tokens_total`

验证后接受的 MTP token 总数。

| 属性 | 值 |
|---|---|
| 类型 | `counter` |
| Prometheus 查询名 | `chitu_mtp_accepted_tokens_total` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | 无 |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |


### 显存

#### `chitu_cuda_total_bytes`

加速器总显存（字节）。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_cuda_total_bytes` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | bytes |
| 范围 | `per_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_cuda_used_bytes`

加速器已用显存（字节），包括 torch 已分配显存、torch 已预留未使用显存以及其他加速器显存。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_cuda_used_bytes` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | bytes |
| 范围 | `per_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_torch_allocated_bytes`

Torch 已分配 GPU 显存（字节）。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_torch_allocated_bytes` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | bytes |
| 范围 | `per_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_torch_reserved_bytes`

Torch 已预留 GPU 显存（字节），包括已分配显存和已预留未使用显存。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_torch_reserved_bytes` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | bytes |
| 范围 | `per_rank` |
| Stdout | 是 |
| Grafana | 是 |


### 请求

#### `chitu_e2e_request_duration_seconds`

从 router 接收请求到 decode 完成的端到端请求延迟。

| 属性 | 值 |
|---|---|
| 类型 | `histogram` |
| Prometheus 查询名 | `chitu_e2e_request_duration_seconds` |
| 标签 | 无 |
| 单位 | seconds |
| 范围 | `whole_service` |
| Stdout | 否 |
| Grafana | 是 |
| Prometheus 还会暴露 | `chitu_e2e_request_duration_seconds_bucket`, `chitu_e2e_request_duration_seconds_count`, `chitu_e2e_request_duration_seconds_sum` |

#### `chitu_time_to_first_token_seconds`

从请求到达到生成首个 token 的时间。

| 属性 | 值 |
|---|---|
| 类型 | `histogram` |
| Prometheus 查询名 | `chitu_time_to_first_token_seconds` |
| 标签 | 无 |
| 单位 | seconds |
| 范围 | `whole_service` |
| Stdout | 否 |
| Grafana | 是 |
| Prometheus 还会暴露 | `chitu_time_to_first_token_seconds_bucket`, `chitu_time_to_first_token_seconds_count`, `chitu_time_to_first_token_seconds_sum` |

#### `chitu_router_pending_requests`

Router 中待处理请求数量。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_router_pending_requests` |
| 标签 | 无 |
| 单位 | 无 |
| 范围 | `whole_service` |
| Stdout | 否 |
| Grafana | 是 |

#### `chitu_active_requests`

活跃流式请求数量。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_active_requests` |
| 标签 | `role` |
| 单位 | 无 |
| 范围 | `per_role` |
| Stdout | 否 |
| Grafana | 是 |

#### `chitu_completed_requests_total`

完成请求总数。

| 属性 | 值 |
|---|---|
| 类型 | `counter` |
| Prometheus 查询名 | `chitu_completed_requests_total` |
| 标签 | `role`, `instance_id`, `rank` |
| 单位 | 无 |
| 范围 | `per_rank` |
| Stdout | 否 |
| Grafana | 是 |

#### `chitu_running_requests`

当前运行中的请求数量。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_running_requests` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | 无 |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_waiting_requests`

当前等待中的请求数量。

| 属性 | 值 |
|---|---|
| 类型 | `gauge` |
| Prometheus 查询名 | `chitu_waiting_requests` |
| 标签 | `rank`, `dp_id`, `instance_id` |
| 单位 | 无 |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |


### 错误和超时

#### `chitu_kv_transfer_failures_total`

KV 传输失败总数。

| 属性 | 值 |
|---|---|
| 类型 | `counter` |
| Prometheus 查询名 | `chitu_kv_transfer_failures_total` |
| 标签 | `role`, `instance_id`, `rank` |
| 单位 | 无 |
| 范围 | `per_rank` |
| Stdout | 否 |
| Grafana | 是 |

#### `chitu_request_timeouts_total`

总超时请求数。

| 属性 | 值 |
|---|---|
| 类型 | `counter` |
| Prometheus 查询名 | `chitu_request_timeouts_total` |
| 标签 | `stage` |
| 单位 | 无 |
| 范围 | `whole_service` |
| Stdout | 否 |
| Grafana | 是 |

## 派生指标

### KV cache

#### `chitu_task_eviction_rate_per_second`

每个 DP rank 的任务驱逐速率。

| 属性 | 值 |
|---|---|
| 表达式 | `rate(chitu_total_task_evictions_total[interval])` |
| 来源指标 | `chitu_total_task_evictions` |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_prefix_cache_hit_rate`

Prefix cache 命中的 prompt token 占总 prompt token 的比例。

| 属性 | 值 |
|---|---|
| 表达式 | `chitu_total_hit_tokens_total / chitu_total_prompt_tokens_total` |
| 来源指标 | `chitu_total_hit_tokens`, `chitu_total_prompt_tokens` |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |


### PD 分离

#### `chitu_pd_stage_latency_quantile`

基于阶段延迟直方图 bucket 的 PD 阶段延迟分位数。

| 属性 | 值 |
|---|---|
| 表达式 | `histogram_quantile(q, sum(rate(chitu_pd_stage_duration_seconds_bucket[interval])) by (le, role, stage))` |
| 来源指标 | `chitu_pd_stage_duration_seconds` |
| 范围 | `per_role` |
| Stdout | 否 |
| Grafana | 是 |

#### `chitu_kv_transfer_speed_bytes_per_second`

KV 传输速度（字节/秒），由传输字节数和传输耗时计算得到。

| 属性 | 值 |
|---|---|
| 表达式 | `sum(rate(chitu_kv_transfer_size_bytes_sum[interval])) by (instance_id, rank) / sum(rate(chitu_kv_transfer_duration_seconds_sum[interval])) by (instance_id, rank)` |
| 来源指标 | `chitu_kv_transfer_size_bytes`, `chitu_kv_transfer_duration_seconds` |
| 范围 | `per_rank` |
| Stdout | 否 |
| Grafana | 是 |


### 吞吐

#### `chitu_prompt_throughput_tokens_per_second`

每个 DP rank 的 prompt token 吞吐。

| 属性 | 值 |
|---|---|
| 表达式 | `rate(chitu_total_prompt_tokens_total[interval])` |
| 来源指标 | `chitu_total_prompt_tokens` |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_generation_throughput_tokens_per_second`

每个 DP rank 的生成 token 吞吐。

| 属性 | 值 |
|---|---|
| 表达式 | `rate(chitu_total_generated_tokens_total[interval])` |
| 来源指标 | `chitu_total_generated_tokens` |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |

#### `chitu_mtp_hit_rate`

MTP 接受 token 占提议 token 的比例。

| 属性 | 值 |
|---|---|
| 表达式 | `rate(chitu_mtp_accepted_tokens_total[interval]) / rate(chitu_mtp_proposed_tokens_total[interval])` |
| 来源指标 | `chitu_mtp_accepted_tokens`, `chitu_mtp_proposed_tokens` |
| 范围 | `per_dp_rank` |
| Stdout | 是 |
| Grafana | 是 |


### 请求

#### `chitu_ttft_latency_quantile`

首 token 延迟分位数。

| 属性 | 值 |
|---|---|
| 表达式 | `histogram_quantile(q, sum(rate(chitu_time_to_first_token_seconds_bucket[interval])) by (le))` |
| 来源指标 | `chitu_time_to_first_token_seconds` |
| 范围 | `whole_service` |
| Stdout | 否 |
| Grafana | 是 |

#### `chitu_e2e_request_latency_quantile`

端到端请求延迟分位数。

| 属性 | 值 |
|---|---|
| 表达式 | `histogram_quantile(q, sum(rate(chitu_e2e_request_duration_seconds_bucket[interval])) by (le))` |
| 来源指标 | `chitu_e2e_request_duration_seconds` |
| 范围 | `whole_service` |
| Stdout | 否 |
| Grafana | 是 |

#### `chitu_completed_request_rate_per_second`

按实例、rank 和角色统计的完成请求速率。

| 属性 | 值 |
|---|---|
| 表达式 | `rate(chitu_completed_requests_total[interval])` |
| 来源指标 | `chitu_completed_requests_total` |
| 范围 | `per_rank` |
| Stdout | 否 |
| Grafana | 是 |


### 错误和超时

#### `chitu_kv_transfer_failure_rate_per_second`

按实例、rank 和角色统计的 KV 传输失败速率。

| 属性 | 值 |
|---|---|
| 表达式 | `rate(chitu_kv_transfer_failures_total[interval])` |
| 来源指标 | `chitu_kv_transfer_failures_total` |
| 范围 | `per_rank` |
| Stdout | 否 |
| Grafana | 是 |

#### `chitu_request_timeout_rate_per_second`

按阶段统计的请求超时速率。

| 属性 | 值 |
|---|---|
| 表达式 | `rate(chitu_request_timeouts_total[interval])` |
| 来源指标 | `chitu_request_timeouts_total` |
| 范围 | `whole_service` |
| Stdout | 否 |
| Grafana | 是 |

## 维护方式

更新 `chitu/metrics/definitions.py` 后运行 `python3 script/generate_metrics_docs.py`。在 CI 中可使用 `python3 script/generate_metrics_docs.py --check` 检查文档是否已同步。生成器只导入指标定义，不会启动 Chitu、Prometheus、Grafana、模型加载或分布式运行时组件。
