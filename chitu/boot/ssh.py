# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
from logging import getLogger

from chitu.boot.appimage_utils import appimage

logger = getLogger(__name__)

# Environment variables to forward to the remote nodes.
_FORWARDED_ENV_VARS = [
    "PATH",
    "LD_LIBRARY_PATH",
    "VIRTUAL_ENV",
    "NCCL_SOCKET_IFNAME",
    "HCCL_SOCKET_IFNAME",
    "NCCL_P2P_LEVEL",
    "NCCL_IB_DISABLE",
    "NCCL_IB_TIMEOUT",
    "NCCL_IB_RETRY_CNT",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_HCA",
    "TP_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_DEBUG",
]


def ssh(cfg, raw_argv, local_run_callback):
    n_nodes = int(cfg.boot.n_nodes)
    node_list = list(cfg.boot.ssh_node_list or [])
    if len(node_list) != n_nodes:
        raise ValueError(
            f"boot.ssh_node_list has {len(node_list)} entries, "
            f"but boot.n_nodes is {n_nodes}. They must be equal."
        )

    if cfg.boot.interactive_node_0:
        raise NotImplementedError(
            "boot.remote_launcher=ssh does not support boot.interactive_node_0 yet"
        )

    # "CHITU_BOOT_IS_IN_NODE" is an internal re-exec marker (set as an
    # environment variable rather than a CLI argument, since extra args are
    # incompatible with Hydra). When it is absent, this is the outer
    # invocation that needs to SSH into each node.
    is_inner_node = bool(os.environ.get("CHITU_BOOT_IS_IN_NODE"))

    if not is_inner_node:
        master_addr = node_list[0]

        # Build the command to run on each node. Forward all original Hydra
        # overrides (raw_argv[1:]) to the inner invocation. The master address
        # is passed via an environment variable so the inner branch can set up
        # the rendezvous.
        inner_argv = [appimage] + list(raw_argv[1:])
        remote_command = " ".join(shlex.quote(str(a)) for a in inner_argv)

        cwd = os.getcwd()

        # Build the environment forwarding prefix.
        env_assignments = []
        for var in _FORWARDED_ENV_VARS:
            value = os.environ.get(var)
            if value is not None:
                env_assignments.append(f"{var}={shlex.quote(value)}")
        # Internal markers passed via environment.
        env_assignments.append("CHITU_BOOT_IS_IN_NODE=1")
        env_assignments.append(f"CHITU_BOOT_MASTER_ADDR={shlex.quote(master_addr)}")

        procs = []
        threads = []
        out_lock = threading.Lock()

        def _forward(node, pipe):
            # Read whole lines and write each atomically so output from
            # different nodes interleaves only at line boundaries. The remote
            # PTY (`-tt`) also puts the local terminal into raw mode, where a
            # bare "\n" only line-feeds without a carriage return. Strip the
            # remote line ending and emit an explicit "\r\n" so lines start at
            # column 0 regardless of terminal mode.
            for line in iter(pipe.readline, ""):
                line = line.rstrip("\r\n")
                with out_lock:
                    sys.stdout.write(f"[{node}] {line}\r\n")
                    sys.stdout.flush()
            pipe.close()

        def _terminate(*_args):
            # Kill all local ssh client processes. Combined with the `-tt`
            # pseudo-TTY allocation below, closing each ssh connection sends a
            # SIGHUP to the remote process group, terminating the remote command
            # tree as well.
            for p in procs:
                if p.poll() is None:
                    try:
                        p.terminate()
                    except ProcessLookupError:
                        pass
            for p in procs:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        p.kill()
                    except ProcessLookupError:
                        pass

        # Ensure remote processes are cleaned up whenever the launcher exits for
        # any reason: normal exit, uncaught exception, or termination signals.
        atexit.register(_terminate)
        signal.signal(signal.SIGINT, lambda s, f: (_terminate(), sys.exit(130)))
        signal.signal(signal.SIGTERM, lambda s, f: (_terminate(), sys.exit(143)))
        signal.signal(signal.SIGHUP, lambda s, f: (_terminate(), sys.exit(129)))

        for node_rank, node in enumerate(node_list):
            per_node_env = list(env_assignments)
            per_node_env.append(f"CHITU_BOOT_NODE_RANK={node_rank}")
            env_prefix = " ".join(per_node_env)

            inner_shell = f"cd {shlex.quote(cwd)}; {env_prefix} {remote_command}"
            # `-tt` forces pseudo-TTY allocation (even when the launcher has no
            # local TTY) so that closing the ssh connection delivers SIGHUP to
            # the remote process group.
            ssh_cmd = ["ssh", "-tt", node, f"bash -c {shlex.quote(inner_shell)}"]
            logger.info(f"Running on node {node}: {ssh_cmd}")
            p = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line buffered
            )
            procs.append(p)
            t = threading.Thread(target=_forward, args=(node, p.stdout), daemon=True)
            t.start()
            threads.append(t)

        exit_code = 0
        for p in procs:
            ret = p.wait()
            if ret != 0:
                exit_code = ret
        for t in threads:
            t.join()
        sys.exit(exit_code)

    logger.debug(f"Running on node {socket.gethostname()}")

    master_addr = os.environ.get("CHITU_BOOT_MASTER_ADDR", node_list[0])

    if n_nodes > 1:
        # Currently we use fixed ports for the rendezvous across all nodes (FIXME).
        master_port = 52000
        rdvz_port = 53000
    else:
        # NOTE: If running on single node, let torchrun pick a random port. It's
        # still sufficiently unique across jobs on this node. See
        # https://docs.pytorch.org/docs/stable/elastic/run.html#stacked-single-node-multi-worker
        master_addr = "127.0.0.1"
        master_port = 0
        rdvz_port = 0
    rdvz_id = "chitu"

    local_run_callback(cfg, raw_argv, master_addr, master_port, rdvz_port, rdvz_id)
