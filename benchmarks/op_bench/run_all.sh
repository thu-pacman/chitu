#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python3 $SCRIPT_DIR/bench_attn.py
python3 $SCRIPT_DIR/bench_cpuinfer_linear.py
python3 $SCRIPT_DIR/bench_cpuinfer_moe_gate.py
python3 $SCRIPT_DIR/bench_cpuinfer_rmsnorm.py
python3 $SCRIPT_DIR/bench_cpuinfer_silu_mul.py
python3 $SCRIPT_DIR/bench_fp4.py
python3 $SCRIPT_DIR/bench_fp8.py
python3 $SCRIPT_DIR/bench_fp8_group_gemm.py
python3 $SCRIPT_DIR/bench_frequency_penalty.py
python3 $SCRIPT_DIR/bench_moe_fuse_gate.py
python3 $SCRIPT_DIR/bench_moe_sum.py
python3 $SCRIPT_DIR/bench_rms_norm.py
python3 $SCRIPT_DIR/bench_rotary.py
python3 $SCRIPT_DIR/bench_silu_and_mul.py
