"""tools/setup_host.py --dashboard: the dashboard image of this version."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import hilspy

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def setup_host(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT / "tools"))  # as when run as a script
    spec = importlib.util.spec_from_file_location("setup_host", ROOT / "tools" / "setup_host.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_image_is_tagged_with_this_version(setup_host: ModuleType) -> None:
    assert setup_host.dashboard_image() == (
        f"ghcr.io/hakuturu583/hilspy-dashboard:{hilspy.__version__}"
    )
    assert setup_host.dashboard_image("latest").endswith(":latest")


def test_dashboard_only_pulls(setup_host: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(setup_host.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(setup_host.subprocess, "run", run)
    monkeypatch.setattr(setup_host, "setup_v4l2", lambda _: pytest.fail("ran the playbook"))
    assert setup_host.main(["--dashboard", "--no-v4l2"]) == 0
    assert calls == [["/usr/bin/docker", "pull", setup_host.dashboard_image()]]
    assert setup_host.main(["--dashboard", "--no-v4l2", "--dashboard-version", "0.1.0"]) == 0
    assert calls[-1][-1].endswith(":0.1.0")
    # --check pulls nothing.
    assert setup_host.main(["--dashboard", "--no-v4l2", "--check"]) == 0
    assert len(calls) == 2


@pytest.mark.parametrize(
    "argv",
    [["--no-v4l2"], ["--dashboard", "--no-v4l2", "--build"], ["--dashboard-version", "1.0.0"]],
)
def test_bad_combinations(setup_host: ModuleType, argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        setup_host.parse_args(argv)
