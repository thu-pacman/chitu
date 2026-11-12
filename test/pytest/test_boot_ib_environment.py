# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import shlex
from types import SimpleNamespace

import pytest


class StopBeforeLaunch(Exception):
    pass


@pytest.fixture
def ib_launch_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDIR", str(tmp_path))
    monkeypatch.setenv("APPIMAGE", "SOURCE")
    monkeypatch.delenv("CHITU_BOOT_IS_IN_NODE", raising=False)
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.setenv("NVSHMEM_HCA_LIST", "mlx5_1:2")
    monkeypatch.setenv("MC_IB_PORT", "2")
    image = tmp_path / "image.sif"
    image.touch()
    return SimpleNamespace(
        boot=SimpleNamespace(
            n_nodes=1,
            ssh_node_list=["node0"],
            interactive_node_0=False,
            extra_docker_args=[],
            extra_apptainer_args=[],
            extra_torchrun_args=[],
            target=["-m", "chitu"],
            container_setup_cmd=None,
            torchrun_wrapper=[],
            on_ready=None,
            container_image=str(image),
            platform="nvidia",
            source_path=None,
            job_name=None,
            relay_args=False,
        )
    )


@pytest.mark.parametrize("launcher", ["docker_run", "apptainer_run"])
def test_container_launch_forwards_hca_selection_and_port(
    monkeypatch, ib_launch_env, launcher, tmp_path
):
    module = importlib.import_module(f"chitu.boot.{launcher}")
    captured = []

    def capture_launch(command, *args, **kwargs):
        captured.extend(command)
        raise StopBeforeLaunch

    monkeypatch.setattr(module.subprocess, "Popen", capture_launch)
    if launcher == "docker_run":
        monkeypatch.setattr(module.shutil, "which", lambda _: None)
        monkeypatch.setattr(module, "get_local_ip", lambda: "127.0.0.1")
        original_mkstemp = module.tempfile.mkstemp
        monkeypatch.setattr(
            module.tempfile,
            "mkstemp",
            lambda **kwargs: original_mkstemp(prefix="hosts-", dir=tmp_path),
        )

    with pytest.raises(StopBeforeLaunch):
        getattr(module, launcher)(
            ib_launch_env,
            ["chitu"],
            "127.0.0.1",
            29500,
            29501,
            "test",
            is_multi_inst=False,
            is_router=False,
            is_master_node=True,
            torchrun_n_nodes=1,
            torchrun_nproc_per_node=1,
        )

    option = "-e" if launcher == "docker_run" else "--env"
    forwarded = [
        value for previous, value in zip(captured, captured[1:]) if previous == option
    ]
    assert "NVSHMEM_HCA_LIST=mlx5_1:2" in forwarded
    assert "MC_IB_PORT=2" in forwarded


def test_ssh_launch_forwards_nvshmem_selection_and_port(monkeypatch, ib_launch_env):
    module = importlib.import_module("chitu.boot.ssh")
    captured = []

    def capture_launch(command, *args, **kwargs):
        captured.extend(command)
        raise StopBeforeLaunch

    monkeypatch.setattr(module.subprocess, "Popen", capture_launch)
    monkeypatch.setattr(module.atexit, "register", lambda _: None)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)

    with pytest.raises(StopBeforeLaunch):
        module.ssh(ib_launch_env, None, ["chitu"], None)

    remote_shell = shlex.split(captured[-1])[2]
    forwarded = shlex.split(remote_shell)
    assert "NVSHMEM_HCA_LIST=mlx5_1:2" in forwarded
    assert "MC_IB_PORT=2" in forwarded
