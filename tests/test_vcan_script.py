"""hilspy/ansible/files/hilspy-vcan, the script vcan.yml and hilspy-vcan.service
run, against a fake `ip` that keeps the interfaces in a JSON file."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")

# A fake iproute2 `ip` for what the script runs: `link show dev N`,
# `-details link show dev N`, `link add dev N type vcan`,
# `link set dev N up|down|mtu M`, `link delete dev N`.
FAKE_IP = """\
import json, os, sys

state_file = os.environ["FAKE_IP_STATE"]
with open(state_file) as f:
    links = json.load(f)
args = sys.argv[1:]
details = args[:1] == ["-details"]
if details:
    args = args[1:]
with open(os.environ["FAKE_IP_LOG"], "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")


def save():
    with open(state_file, "w") as f:
        json.dump(links, f)


assert args[0] == "link" and "dev" in args, args
name = args[args.index("dev") + 1]
verb = args[1]
if verb == "show":
    link = links.get(name)
    if link is None:
        print(f'Device "{name}" does not exist.', file=sys.stderr)
        sys.exit(1)
    flags = "NOARP,UP,LOWER_UP" if link["up"] else "NOARP"
    print(f"3: {name}: <{flags}> mtu {link['mtu']} qdisc noqueue state UNKNOWN mode DEFAULT")
    print("    link/can  promiscuity 0 minmtu 0 maxmtu 0")
    if details:
        kind = "vcan" if link["kind"] == "vcan" else "can state STOPPED restart-ms 0"
        print(f"    {kind} addrgenmode none numtxqueues 1")
elif verb == "add":
    links[name] = {"kind": "vcan", "mtu": 16, "up": False}
    save()
elif verb == "delete":
    del links[name]
    save()
elif verb == "set":
    link = links[name]
    rest = args[args.index("dev") + 2:]
    if rest == ["up"]:
        link["up"] = True
    elif rest == ["down"]:
        link["up"] = False
    elif rest[0] == "mtu":
        if link["up"]:
            print("RTNETLINK answers: Device or resource busy", file=sys.stderr)
            sys.exit(2)
        link["mtu"] = int(rest[1])
    save()
else:
    sys.exit(f"unexpected: {args}")
"""


class Host:
    def __init__(self, tmp_path: Path) -> None:
        self.state = tmp_path / "links.json"
        self.log = tmp_path / "ip.log"
        self.config = tmp_path / "vcan-interfaces"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        ip = bin_dir / "ip"
        ip.write_text(f"#!{sys.executable}\n{FAKE_IP}")
        ip.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_IP_STATE": str(self.state),
            "FAKE_IP_LOG": str(self.log),
            "HILSPY_VCAN_CONFIG": str(self.config),
        }
        self.links({})
        self.log.write_text("")

    def links(self, links: dict[str, Any] | None = None) -> dict[str, Any]:
        if links is not None:
            self.state.write_text(json.dumps(links))
        result: dict[str, Any] = json.loads(self.state.read_text())
        return result

    def changes(self) -> list[str]:
        """The `ip` calls that changed something."""
        return [line for line in self.log.read_text().splitlines() if " show " not in line]

    def run(self, verb: str, listed: str) -> subprocess.CompletedProcess[str]:
        self.config.write_text(listed)
        script = resources.files("hilspy").joinpath("ansible/files/hilspy-vcan")
        return subprocess.run(
            ["sh", str(script), verb], env=self.env, capture_output=True, text=True
        )


@pytest.fixture
def host(tmp_path: Path) -> Host:
    return Host(tmp_path)


def test_up_creates_the_interfaces_once(host: Host) -> None:
    result = host.run("up", "vcan0 72\nvcan1 16\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["changed"]
    assert host.links() == {
        "vcan0": {"kind": "vcan", "mtu": 72, "up": True},
        "vcan1": {"kind": "vcan", "mtu": 16, "up": True},
    }
    host.log.write_text("")
    again = host.run("up", "vcan0 72\nvcan1 16\n")
    assert again.returncode == 0 and again.stdout == ""
    assert host.changes() == []


def test_up_puts_back_what_changed_since(host: Host) -> None:
    # Deleted, taken down, given another MTU while up: what an unchanged
    # unit, already active, would not have repaired.
    host.links({"vcan1": {"kind": "vcan", "mtu": 16, "up": False}})
    host.links({**host.links(), "vcan2": {"kind": "vcan", "mtu": 16, "up": True}})
    result = host.run("up", "vcan0 72\nvcan1 16\nvcan2 72\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["changed"]
    assert host.links() == {
        "vcan0": {"kind": "vcan", "mtu": 72, "up": True},
        "vcan1": {"kind": "vcan", "mtu": 16, "up": True},
        "vcan2": {"kind": "vcan", "mtu": 72, "up": True},
    }


def test_up_leaves_an_interface_of_another_kind_alone(host: Host) -> None:
    adapter = {"kind": "can", "mtu": 16, "up": True}
    host.links({"can0": adapter})
    result = host.run("up", "vcan0 72\ncan0 72\n")
    assert result.returncode == 1
    assert "can0 is an interface of another kind" in result.stderr
    # Refused before anything changed: not even vcan0 was created.
    assert host.links() == {"can0": adapter}
    assert host.changes() == []


def test_down_deletes_only_vcan_interfaces(host: Host) -> None:
    adapter = {"kind": "can", "mtu": 16, "up": True}
    host.links({"vcan0": {"kind": "vcan", "mtu": 72, "up": True}, "can0": adapter})
    result = host.run("down", "vcan0 72\ncan0 16\nvcan9 16\n")
    assert result.returncode == 0, result.stderr
    assert host.links() == {"can0": adapter}


def test_usage(host: Host) -> None:
    assert host.run("restart", "").returncode == 2
