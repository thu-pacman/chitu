# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the per-job port blocks used by the Slurm launcher.

The ports of a job must stay outside the kernel's local port range
(`net.ipv4.ip_local_port_range`, 32768-60999 by default), where they can be
occupied by an outbound connection of any process on the same node, and
concurrent jobs (which get consecutive Slurm job IDs) must not share ports.
"""

import pytest

from chitu.boot.tcp_ip import (
    MAX_INSTS_PER_JOB,
    PORT_BLOCK_BASE,
    PORT_BLOCK_COUNT,
    PORT_BLOCK_LIMIT,
    PORTS_PER_JOB,
    job_port_base,
)

#: The kernel's local port range on the CI hosts, see `sysctl net.ipv4.ip_local_port_range`.
KERNEL_PORT_RANGE = (32768, 60999)

#: Port ranges used by other parts of chitu on the same node: the KV cache
#: transfer engines (see `chitu/distributed/pd_disaggregation/kv_transfer/mooncake/transfer_engine.py`)
#: and the random range CI cases draw `serve.port` from, see `ci/platforms/h20/pd_test.yml`.
OTHER_PORT_RANGES = ((10000, 10032), (12000, 13000), (20000, 32768))


def test_port_blocks_are_outside_kernel_port_range():
    assert PORT_BLOCK_LIMIT <= KERNEL_PORT_RANGE[0]
    last_port = job_port_base(PORT_BLOCK_COUNT - 1) + PORTS_PER_JOB
    assert last_port <= PORT_BLOCK_LIMIT


@pytest.mark.parametrize("low,high", OTHER_PORT_RANGES + (KERNEL_PORT_RANGE,), ids=str)
def test_port_blocks_do_not_overlap_other_port_ranges(low, high):
    # A job block must either end before the other range, or start after it.
    assert PORT_BLOCK_LIMIT <= low or PORT_BLOCK_BASE >= high


@pytest.mark.parametrize("slurm_job_id", [0, 1, 123, 886453, 886512, 887689, 10**9])
def test_job_port_base_is_deterministic_and_in_range(slurm_job_id):
    base = job_port_base(slurm_job_id)
    assert base == job_port_base(slurm_job_id)
    assert PORT_BLOCK_BASE <= base < PORT_BLOCK_LIMIT
    assert (base - PORT_BLOCK_BASE) % PORTS_PER_JOB == 0


def test_concurrent_jobs_get_disjoint_blocks():
    # Slurm hands out consecutive job IDs to jobs scheduled at the same time,
    # so consecutive jobs must never share a port on a node.
    for slurm_job_id in range(200):
        ports = set(
            range(
                job_port_base(slurm_job_id),
                job_port_base(slurm_job_id) + PORTS_PER_JOB,
            )
        )
        next_ports = set(
            range(
                job_port_base(slurm_job_id + 1),
                job_port_base(slurm_job_id + 1) + PORTS_PER_JOB,
            )
        )
        assert ports.isdisjoint(next_ports)


@pytest.mark.parametrize("n_insts", [1, 2, 4, 8, MAX_INSTS_PER_JOB])
def test_block_has_room_for_coordinator_and_all_instances(n_insts):
    # One coordinator port, plus one master port and one rendezvous port per
    # instance.
    ports = {job_port_base(886453)}
    ports |= {job_port_base(886453) + 1 + inst_id for inst_id in range(n_insts)}
    ports |= {
        job_port_base(886453) + 1 + n_insts + inst_id for inst_id in range(n_insts)
    }
    assert len(ports) == 1 + 2 * n_insts
    assert max(ports) < job_port_base(886453) + PORTS_PER_JOB


def test_block_is_too_small_for_more_instances_than_supported():
    # Documents why a port block holds exactly `PORTS_PER_JOB` ports: one
    # coordinator port, plus one master port and one rendezvous port per
    # instance, must fit into the block.
    n_insts = MAX_INSTS_PER_JOB + 1
    base = job_port_base(886453)
    assert base + 1 + 2 * n_insts - 1 >= base + PORTS_PER_JOB
