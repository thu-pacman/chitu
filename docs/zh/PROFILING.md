# 性能分析

赤兔内置 profile 功能，可以在服务运行过程中按需触发性能分析，不需要修改代码或重启服务。

此功能支持两类分析：

| 类别 | 用途 | 产出文件 | 查看工具 |
|---|---|---|---|
| Torch Profiler | 分析 CPU/GPU 算子耗时 | `.pt.trace.json.gz` | `chrome://tracing` 或 [Perfetto](https://ui.perfetto.dev) |
| Memory Tracking | 记录 CUDA 显存分配生命周期，并在 OOM 时自动导出快照 | `.memory_snapshot.pickle` | [pytorch.org/memory_viz](https://pytorch.org/memory_viz) |

两者可以独立使用，也可以同时开启。

## Torch Profiler

### 启动方式

Torch Profiler 通过 HTTP API 在运行中的服务上开启或停止。

**开始采集：**

```bash
curl -X POST http://<host>:30000/profile/start \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "my_run",
    "num_steps": 20,
    "activities": ["CPU", "GPU"]
  }'
```

**停止采集：**

```bash
curl -X POST http://<host>:30000/profile/stop
```

### `/profile/start` 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `output_dir` | string | `"trace/chitu"` | 输出目录。相对路径会写入 `CHITU_TORCH_PROFILER_OUTPUT_ROOT` 下 |
| `activities` | list | `["CPU", "GPU"]` | 采集类型，可选 `"CPU"`、`"GPU"`、`"MEM"` |
| `start_step` | int | `0` | 跳过前 N 步后开始采集 |
| `num_steps` | int | `10` | 自动停止前采集的步数 |
| `with_stack` | bool | `false` | 是否记录 Python 调用栈 |
| `profile_by_stage` | bool | `false` | 是否按 Prefill/Decode 阶段分别采集 |
| `profile_memory` | bool | `false` | 是否等同于在 `activities` 中加入 `"MEM"` |
| `memory_max_entries` | int | `100000` | `MEM` 模式下的环形缓冲区大小 |
| `pd_stage` | string 或 null | `null` | PD 分离部署中的目标阶段，可选 `"prefill"`、`"decode"`、`"all"` |

### 示例

下面的脚本保留完整流程，但省略了集群、容器和模型参数。请先按正常方式启动赤兔服务，再运行这个脚本触发一次 profile。

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:30000}"
RUN_ID="${RUN_ID:-profile_$(date +%Y%m%d_%H%M%S)}"

curl -fsS -X POST "$BASE_URL/ping" >/dev/null

curl -fsS -X POST "$BASE_URL/profile/start" \
  -H "Content-Type: application/json" \
  -d "{\"output_dir\": \"$RUN_ID\", \"num_steps\": 20}"

python3 benchmarks/benchmark_serving.py \
  --model "<model-name>" \
  --num-requests 16 \
  --input-len 1024 \
  --output-len 128 \
  --base-url "$BASE_URL"

curl -fsS -X POST "$BASE_URL/profile/stop"
```

### 输出文件

每个 rank 会产生一个 trace 文件，例如：

```text
trace/chitu/my_run/20260330_103421.rank_0.node-020.pt.trace.json.gz
trace/chitu/my_run/20260330_103421.rank_1.node-020.pt.trace.json.gz
```

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHITU_TORCH_PROFILER_OUTPUT_ROOT` | `trace/chitu` | Torch Profiler trace 输出根目录 |

## Memory Tracking

### 启动方式

Memory Tracking 需要在服务启动前设置环境变量，不能在运行中开启：

```bash
export CHITU_MEM_TRACK=1
export CHITU_MEM_TRACK_MAX_ENTRIES=5000000
export CHITU_MEM_TRACK_SNAPSHOT_DIR=/path/to/snapshots
```

设置后启动服务，显存分配记录会从模型加载前开始。

### 导出 snapshot

通过 HTTP API 触发所有 rank 导出快照：

```bash
curl -X POST http://<host>:30000/profile/dump_memory
```

此请求应在推理请求仍在处理时发送。如果没有推理请求在跑，可能只有 rank 0 导出快照，因为其他 rank 通过推理调度通道接收 dump 指令。

### 示例

下面的脚本展示一次最小 Memory Tracking 流程。它假设服务已经用 `CHITU_MEM_TRACK=1` 启动。

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:30000}"

python3 benchmarks/benchmark_serving.py \
  --model "<model-name>" \
  --num-requests 64 \
  --input-len 1024 \
  --output-len 128 \
  --base-url "$BASE_URL" &
BENCH_PID=$!

sleep 10
curl -fsS -X POST "$BASE_URL/profile/dump_memory"

wait "$BENCH_PID"
```

进程退出时也会自动导出一份 snapshot；CUDA OOM 时会自动导出带 `OOM` 标记的 snapshot。

### 输出文件

```text
20260330_103421.rank_0.node-020-api.memory_snapshot.pickle
20260330_103421.rank_0.node-020-OOM.memory_snapshot.pickle
```

文件名中的 tag 含义：

| Tag | 触发方式 |
|---|---|
| `api` | 通过 `/profile/dump_memory` API 触发 |
| `OOM` | CUDA OOM 时自动触发 |
| `atexit` | 进程退出时自动触发 |

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHITU_MEM_TRACK` | （未设置） | 设为 `1` 启用启动时显存跟踪 |
| `CHITU_MEM_TRACK_MAX_ENTRIES` | `1000000` | 环形缓冲区大小。越大记录越完整，但占用更多主机内存 |
| `CHITU_MEM_TRACK_SNAPSHOT_DIR` | `trace/chitu/mem_track` | snapshot 输出目录 |

