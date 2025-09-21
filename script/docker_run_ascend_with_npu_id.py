#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# Launch a docker container on Ascend platform with specific NPU IDs set in `ASCEND_RT_VISIBLE_DEVICES`.
# Can be used together with `grun`, which sets `ASCEND_RT_VISIBLE_DEVICES`.
#
# Example:
# ```
# grun <npu_count> ./script/docker_run_ascend_with_npu_id.py <other_docker_run_args_and_commands>...
# ```

import os
import sys

if not "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
    print(f"{sys.argv[0]} is supposed to be used with ASCEND_RT_VISIBLE_DEVICES set.")
    exit(1)


npu_ids = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
pwd = os.path.abspath("./")

cmd = "docker"
args = [
    "docker",
    "run",
    "--device",
    "/dev/davinci_manager",
    "--device",
    "/dev/devmm_svm",
    "--device",
    "/dev/hisi_hdc",
    "-v",
    "/usr/local/dcmi:/usr/local/dcmi",
    "-v",
    "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi",
    "-v",
    "/usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/",
    "-v",
    "/usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info",
    "-v",
    "/etc/ascend_install.info:/etc/ascend_install.info",
    "-v",
    f"{pwd}:/tmp/chitu",
    "-w",
    "/workspace/chitu",
]
for npu_id in npu_ids:
    args += ["--device", f"/dev/davinci{npu_id}"]
for arg in sys.argv[1:]:
    args.append(arg)

human_readable_cmd = " ".join(["'" + arg + "'" for arg in args])
print(f"Running: {human_readable_cmd}", flush=True)
os.execlp(cmd, *args)
