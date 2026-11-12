# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

import pytest

from chitu.distributed import infiniband


@pytest.fixture(autouse=True)
def ib_environment(monkeypatch):
    for name in (
        "NCCL_IB_HCA",
        "NVSHMEM_HCA_LIST",
        "MC_IB_PORT",
        "MOONCAKE_PROTOCOL",
        "MOONCAKE_MOCK_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    infiniband.detect_ib_devices.cache_clear()
    infiniband.get_recommended_ib_envs.cache_clear()
    yield
    infiniband.detect_ib_devices.cache_clear()
    infiniband.get_recommended_ib_envs.cache_clear()


@pytest.fixture
def ibstat(monkeypatch, tmp_path):
    monkeypatch.setattr(infiniband, "_IB_SYSFS_PATH", tmp_path, raising=False)
    for name, state in [
        ("mlx5_0", "4: ACTIVE"),
        ("mlx5_1", "4: ACTIVE"),
        ("mlx5_10", "4: ACTIVE"),
        ("mlx5_3", "4: ACTIVE"),
        ("mlx5_4", "1: DOWN"),
        ("mlx5_5", "4: ACTIVE"),
    ]:
        port = tmp_path / name / "ports" / "1"
        port.mkdir(parents=True)
        (port / "state").write_text(state + "\n")
    output = "\n".join(
        f"CA '{name}'\n    Number of ports: 1\n    Port 1:\n"
        f"        State: {state}\n        Physical state: LinkUp\n"
        f"        Rate: {rate}\n"
        for name, state, rate in [
            ("mlx5_0", "Active", 400),
            ("mlx5_1", "Active", 400),
            ("mlx5_10", "Active", 400),
            ("mlx5_3", "Active", 400),
            ("mlx5_4", "Down", 400),
            ("mlx5_5", "Active", 200),
        ]
    )
    monkeypatch.setattr(infiniband.subprocess, "check_output", lambda *a, **kw: output)


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("mlx5_0,mlx5_1,mlx5_5", "mlx5_0,mlx5_1,mlx5_10,mlx5_5"),
        ("=mlx5_0,mlx5_1,mlx5_5", "mlx5_0,mlx5_1,mlx5_5"),
        ("mlx5_1", "mlx5_1,mlx5_10"),
        ("^mlx5_1,mlx5_3", "mlx5_0,mlx5_5"),
        ("^=mlx5_1,mlx5_3", "mlx5_0,mlx5_10,mlx5_5"),
        ("=mlx5_5", "mlx5_5"),
        ("=mlx5_0:1,mlx5_1:2", "mlx5_0"),
        ("=mlx5_0:1:0:0", "mlx5_0"),
        ("=mlx5_0::0:0", "mlx5_0"),
    ],
)
def test_mooncake_resolves_nccl_selectors(monkeypatch, ibstat, selection, expected):
    monkeypatch.setenv("NCCL_IB_HCA", selection)

    assert infiniband.detect_mooncake_ib_devices() == expected


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("mlx5_1", "mlx5_1"),
        ("mlx5_0,mlx5_5", "mlx5_0,mlx5_5"),
        ("^mlx5_1,mlx5_3", "mlx5_0,mlx5_10,mlx5_5"),
    ],
)
def test_mooncake_resolves_nvshmem_selectors(monkeypatch, ibstat, selection, expected):
    monkeypatch.setenv("NVSHMEM_HCA_LIST", selection)

    assert infiniband.detect_mooncake_ib_devices() == expected


def test_mooncake_prefers_nccl_setting(monkeypatch, ibstat):
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_0")
    monkeypatch.setenv("NVSHMEM_HCA_LIST", "mlx5_3")

    assert infiniband.detect_mooncake_ib_devices() == "mlx5_0"


@pytest.mark.parametrize("selection", ["mlx5_99", "=mlx5_4", "^mlx5_"])
def test_no_matching_device_does_not_fall_back(monkeypatch, ibstat, selection):
    monkeypatch.setenv("NCCL_IB_HCA", selection)

    with pytest.raises(ValueError, match="NCCL_IB_HCA"):
        infiniband.detect_mooncake_ib_devices()


@pytest.mark.parametrize("selection", ["=", "^", ",", "mlx5_0:bad", "mlx5_0:1:2:3:4"])
def test_malformed_selection_is_reported(monkeypatch, ibstat, selection):
    monkeypatch.setenv("NCCL_IB_HCA", selection)

    with pytest.raises(ValueError, match="NCCL_IB_HCA"):
        infiniband.detect_mooncake_ib_devices()


def test_unset_selection_keeps_fastest_device_detection(ibstat):
    assert infiniband.detect_ib_devices() == "mlx5_0,mlx5_1,mlx5_10,mlx5_3"
    assert infiniband.detect_mooncake_ib_devices() == "mlx5_0,mlx5_1,mlx5_10,mlx5_3"


def test_empty_nccl_selection_uses_nvshmem(monkeypatch, ibstat):
    monkeypatch.setenv("NCCL_IB_HCA", "")
    monkeypatch.setenv("NVSHMEM_HCA_LIST", "mlx5_5")

    assert infiniband.detect_mooncake_ib_devices() == "mlx5_5"


def test_mooncake_selection_does_not_change_unfiltered_detection(monkeypatch, ibstat):
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_5")

    assert infiniband.detect_mooncake_ib_devices() == "mlx5_5"
    assert infiniband.detect_ib_devices() == "mlx5_0,mlx5_1,mlx5_10,mlx5_3"


@pytest.mark.parametrize(
    "env_name,selection,expected",
    [
        ("NVSHMEM_HCA_LIST", "mlx5_1", "mlx5_1"),
        ("NVSHMEM_HCA_LIST", "mlx5_5", "mlx5_5"),
        ("NVSHMEM_HCA_LIST", "^mlx5_1,mlx5_3", "mlx5_0,mlx5_10,mlx5_5"),
        ("NCCL_IB_HCA", "=mlx5_1", "mlx5_1"),
        ("NCCL_IB_HCA", "^mlx5_1,mlx5_3", "mlx5_0,mlx5_5"),
    ],
)
def test_automatic_env_setup_preserves_explicit_selection(
    monkeypatch, ibstat, env_name, selection, expected
):
    monkeypatch.setenv(env_name, selection)
    monkeypatch.setattr(infiniband, "detect_ib_network_interface", lambda: None)

    infiniband.auto_set_ib_envs()

    assert infiniband.detect_mooncake_ib_devices() == expected
    assert os.environ[env_name] == selection


def test_automatic_nccl_selection_matches_names_exactly(monkeypatch, ibstat):
    monkeypatch.setattr(infiniband, "detect_ib_network_interface", lambda: None)

    infiniband.auto_set_ib_envs()

    assert os.environ["NCCL_IB_HCA"] == "=mlx5_0,mlx5_1,mlx5_10,mlx5_3"
    assert os.environ["NVSHMEM_HCA_LIST"] == "mlx5_0,mlx5_1,mlx5_10,mlx5_3"


@pytest.mark.parametrize(
    "error", [FileNotFoundError(), subprocess.TimeoutExpired("ibstat", 10)]
)
def test_explicit_selection_requires_device_discovery(monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(infiniband.subprocess, "check_output", unavailable)
    assert infiniband.detect_ib_devices() is None


@pytest.mark.parametrize("selection", ["=mlx5_0", "mlx5_0", "^=mlx5_3"])
def test_explicit_selection_does_not_require_ibstat(monkeypatch, ibstat, selection):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("ibstat")

    monkeypatch.setattr(infiniband.subprocess, "check_output", unavailable)
    monkeypatch.setenv("NCCL_IB_HCA", selection)

    assert "mlx5_0" in infiniband.detect_mooncake_ib_devices().split(",")


def test_selection_checks_the_actual_mooncake_port(monkeypatch, ibstat):
    root = infiniband._IB_SYSFS_PATH
    (root / "mlx5_0/ports/1/state").write_text("1: DOWN\n")
    second_port = root / "mlx5_0/ports/2"
    second_port.mkdir()
    (second_port / "state").write_text("4: ACTIVE\n")
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_0")

    with pytest.raises(ValueError, match="NCCL_IB_HCA"):
        infiniband.detect_mooncake_ib_devices()

    monkeypatch.setenv("MC_IB_PORT", "2")
    assert infiniband.detect_mooncake_ib_devices() == "mlx5_0"


@pytest.mark.parametrize("port", ["0", "256", "bad", ""])
def test_invalid_mooncake_port_is_reported(monkeypatch, ibstat, port):
    monkeypatch.setenv("MC_IB_PORT", port)
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_0")

    with pytest.raises(ValueError, match="MC_IB_PORT"):
        infiniband.detect_mooncake_ib_devices()


def test_no_visible_rdma_devices_does_not_fall_back(monkeypatch, tmp_path):
    monkeypatch.setattr(infiniband, "_IB_SYSFS_PATH", tmp_path, raising=False)
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_0")

    with pytest.raises(ValueError, match="NCCL_IB_HCA"):
        infiniband.detect_mooncake_ib_devices()


@pytest.mark.parametrize(
    "protocol,ascend,mock_mode,expected",
    [
        ("rdma", False, False, "mlx5_0,mlx5_5"),
        ("tcp", False, False, None),
        ("rdma", True, False, None),
        ("hccl", False, False, None),
        ("rdma", False, True, None),
    ],
)
def test_kv_manager_passes_selected_devices_to_mooncake(
    monkeypatch, ibstat, protocol, ascend, mock_mode, expected
):
    from types import SimpleNamespace
    from chitu.distributed.pd_disaggregation.kv_transfer import base

    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_0,mlx5_5")
    monkeypatch.setenv("MOONCAKE_PROTOCOL", protocol)
    monkeypatch.setenv("MOONCAKE_MOCK_MODE", "1" if mock_mode else "0")
    monkeypatch.setattr(base, "is_ascend", lambda: ascend, raising=False)
    if expected is None:

        def no_ibstat(*args, **kwargs):
            raise FileNotFoundError("ibstat")

        monkeypatch.setattr(infiniband.subprocess, "check_output", no_ibstat)
        monkeypatch.setattr(
            infiniband, "_IB_SYSFS_PATH", infiniband._IB_SYSFS_PATH / "absent"
        )
    monkeypatch.setattr(
        base,
        "get_global_args",
        lambda: SimpleNamespace(
            multi_inst=SimpleNamespace(
                inst_id=0,
                pd_disaggregation=SimpleNamespace(kv_transfer=SimpleNamespace()),
            ),
            boot=SimpleNamespace(restart_instance_id=None),
        ),
    )
    monkeypatch.setattr(base, "get_local_ip", lambda: "127.0.0.1")
    captured = {}

    class StopBeforeRdmaStartup(Exception):
        pass

    def capture_engine(**kwargs):
        captured.update(kwargs)
        raise StopBeforeRdmaStartup

    monkeypatch.setattr(base, "MooncakeTransferEngine", capture_engine)

    with pytest.raises(StopBeforeRdmaStartup):
        base.KVManagerBase(base.DisaggregationMode.DECODE)

    assert captured == {"hostname": "127.0.0.1", "ib_device": expected}
