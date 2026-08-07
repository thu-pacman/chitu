# Profiling

Chitu has built-in profiling support that can be triggered on demand while a service is running. It does not require code changes or a service restart.

Two profiling modes are supported:

| Category | Purpose | Output Files | Viewer |
|---|---|---|---|
| Torch Profiler | Analyze CPU/GPU operator latency | `.pt.trace.json.gz` | `chrome://tracing` or [Perfetto](https://ui.perfetto.dev) |
| Memory Tracking | Record the CUDA memory allocation lifecycle and automatically dump snapshots on OOM | `.memory_snapshot.pickle` | [pytorch.org/memory_viz](https://pytorch.org/memory_viz) |

They can be used independently or enabled together.

## Torch Profiler

### How to start

Torch Profiler is started and stopped through HTTP APIs on a running service.

**Start profiling:**

```bash
curl -X POST http://<host>:30000/profile/start \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "my_run",
    "num_steps": 20,
    "activities": ["CPU", "GPU"]
  }'
```

**Stop profiling:**

```bash
curl -X POST http://<host>:30000/profile/stop
```

### `/profile/start` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_dir` | string | `"trace/chitu"` | Output directory. Relative paths are written under `CHITU_TORCH_PROFILER_OUTPUT_ROOT` |
| `activities` | list | `["CPU", "GPU"]` | Activity types. Values can include `"CPU"`, `"GPU"`, and `"MEM"` |
| `start_step` | int | `0` | Skip the first N steps before collecting data |
| `num_steps` | int | `10` | Number of steps to collect before automatic stop |
| `with_stack` | bool | `false` | Whether to record Python stack traces |
| `profile_by_stage` | bool | `false` | Whether to collect Prefill/Decode stages separately |
| `profile_memory` | bool | `false` | Equivalent to adding `"MEM"` to `activities` |
| `memory_max_entries` | int | `100000` | Ring-buffer size for `MEM` mode |
| `pd_stage` | string or null | `null` | Target stage in PD disaggregated serving. Values can be `"prefill"`, `"decode"`, or `"all"` |

### Example

The script below keeps the full workflow but omits cluster, container, and model-specific arguments. Start a Chitu service normally first, then run this script to collect one profile.

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

### Output files

Each rank produces one trace file, for example:

```text
trace/chitu/my_run/20260330_103421.rank_0.node-020.pt.trace.json.gz
trace/chitu/my_run/20260330_103421.rank_1.node-020.pt.trace.json.gz
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `CHITU_TORCH_PROFILER_OUTPUT_ROOT` | `trace/chitu` | Root directory for Torch Profiler trace files |

## Memory Tracking

### How to start

Memory Tracking must be enabled with environment variables before starting the service. It cannot be enabled after the service has started.

```bash
export CHITU_MEM_TRACK=1
export CHITU_MEM_TRACK_MAX_ENTRIES=5000000
export CHITU_MEM_TRACK_SNAPSHOT_DIR=/path/to/snapshots
```

After these variables are set, start the service. CUDA memory allocation history starts before model loading.

### Dump snapshots

Use the HTTP API to trigger snapshot dumping on all ranks:

```bash
curl -X POST http://<host>:30000/profile/dump_memory
```

Send this request while inference requests are still being processed. If no inference request is running, only rank 0 may dump a snapshot because the other ranks receive the dump command through the inference scheduling channel.

### Example

The script below shows a minimal Memory Tracking workflow. It assumes the service has already been started with `CHITU_MEM_TRACK=1`.

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

The process also dumps a snapshot on exit. CUDA OOM automatically dumps a snapshot tagged with `OOM`.

### Output files

```text
20260330_103421.rank_0.node-020-api.memory_snapshot.pickle
20260330_103421.rank_0.node-020-OOM.memory_snapshot.pickle
```

Tags in file names:

| Tag | Trigger |
|---|---|
| `api` | Triggered by the `/profile/dump_memory` API |
| `OOM` | Automatically triggered by CUDA OOM |
| `atexit` | Automatically triggered when the process exits |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `CHITU_MEM_TRACK` | (unset) | Set to `1` to enable startup-time memory tracking |
| `CHITU_MEM_TRACK_MAX_ENTRIES` | `1000000` | Ring-buffer size. Larger values keep more complete history but use more host memory |
| `CHITU_MEM_TRACK_SNAPSHOT_DIR` | `trace/chitu/mem_track` | Snapshot output directory |

