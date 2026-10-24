"""Prepare this host for hilspy: ``v4l2://`` cameras and the metrics dashboard.

Runs ``tools/ansible/v4l2loopback.yml`` on localhost through ansible-runner::

    uv run --group setup tools/setup_host.py                 # /dev/video10
    uv run --group setup tools/setup_host.py --devices 2 --build
    uv run --group setup tools/setup_host.py --dashboard     # and docker pull
    uv run tools/setup_host.py --dashboard --no-v4l2         # only docker pull

It installs GStreamer and v4l2loopback (Debian / Ubuntu), creates the
devices now and at every boot, adds you to the ``video`` group and, with
``--build``, rebuilds this checkout with the ``v4l2`` feature. ``sudo`` is
used for the system changes; its password is asked for when needed.

``--dashboard`` pulls the dashboard image (Prometheus receiving hilspy's
OTLP metrics, Perses showing them) tagged with this checkout's version, so
the dashboard matches the metrics this hilspy exports.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PLAYBOOK = ROOT / "tools" / "ansible" / "v4l2loopback.yml"
#: Built and pushed for every release by .github/workflows/dashboard.yml.
DASHBOARD_IMAGE = "ghcr.io/hakuturu583/hilspy-dashboard"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--devices", type=int, default=1, help="number of cameras (default 1)")
    p.add_argument("--first", type=int, default=10, help="first /dev/videoN (default 10)")
    p.add_argument("--label", default="hilspy camera", help="card label prefix")
    p.add_argument(
        "--no-exclusive-caps",
        action="store_true",
        help="advertise capture and output at once (exclusive_caps=0); browsers then miss them",
    )
    p.add_argument("--build", action="store_true", help="rebuild this checkout with `v4l2`")
    p.add_argument("--check", action="store_true", help="show what would change, change nothing")
    p.add_argument("--skip-tags", help="ansible tags to skip: packages, module")
    p.add_argument(
        "--dashboard",
        action="store_true",
        help=f"docker pull {DASHBOARD_IMAGE}:<this checkout's version>",
    )
    p.add_argument(
        "--dashboard-version", help="pull this version's dashboard instead (e.g. 0.15.0, latest)"
    )
    p.add_argument("--no-v4l2", action="store_true", help="skip the v4l2loopback setup")
    args = p.parse_args(argv)
    if not 1 <= args.devices <= 8 or args.first < 0:
        p.error("--devices is 1..8 and --first a device number")
    if args.no_v4l2 and not args.dashboard:
        p.error("--no-v4l2 leaves nothing to do without --dashboard")
    if args.no_v4l2 and args.build:
        p.error("--build needs the v4l2 setup (GStreamer), drop --no-v4l2")
    if args.dashboard_version and not args.dashboard:
        p.error("--dashboard-version goes with --dashboard")
    return args


def extravars(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "v4l2loopback_count": args.devices,
        "v4l2loopback_first": args.first,
        "v4l2loopback_label": args.label,
        "v4l2loopback_exclusive_caps": not args.no_exclusive_caps,
        "hilspy_user": os.environ.get("SUDO_USER") or getpass.getuser(),
        "hilspy_build": args.build,
    }


def build() -> int:
    """Rebuild this checkout's extension with the `v4l2` feature."""
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        sys.exit("--build needs uv (https://docs.astral.sh/uv/)")
    env = {**os.environ, "MATURIN_PEP517_ARGS": "--features v4l2"}
    return subprocess.run(
        [uv, "sync", "--reinstall-package", "hilspy"], cwd=ROOT, env=env
    ).returncode


def version() -> str:
    """This checkout's hilspy version (pyproject.toml), the dashboard's tag."""
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    if not match:
        sys.exit("no version in pyproject.toml")
    return match.group(1)


def dashboard_image(tag: str | None = None) -> str:
    return f"{DASHBOARD_IMAGE}:{tag or version()}"


def pull_dashboard(image: str, check: bool = False) -> int:
    """``docker pull`` the dashboard and say how to run it."""
    if check:
        print(f"would pull {image}")
        return 0
    docker = shutil.which("docker")
    if not docker:
        sys.exit("--dashboard needs docker (https://docs.docker.com/engine/install/)")
    rc = subprocess.run([docker, "pull", image]).returncode
    if rc != 0:
        print(
            f"could not pull {image}: is that version released? "
            "--dashboard-version picks another (latest: the newest release)",
            file=sys.stderr,
        )
        return rc
    print(
        "\ndashboard ready:\n"
        "  docker run -d --name hilspy-dashboard -p 9090:9090 -p 8080:8080 \\\n"
        f"    -v hilspy-prometheus:/prometheus {image}\n"
        "  export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:9090/api/v1/otlp\n"
        "  open http://localhost:8080/projects/hilspy/dashboards/hilspy"
    )
    return 0


def become_password() -> str | None:
    """The sudo password, or None when root or sudo needs none."""
    if os.geteuid() == 0:
        return None
    ok = subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
    return None if ok else getpass.getpass("sudo password: ")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.no_v4l2 and (rc := setup_v4l2(args)) != 0:
        return rc
    if args.dashboard:
        return pull_dashboard(dashboard_image(args.dashboard_version), check=args.check)
    return 0


def setup_v4l2(args: argparse.Namespace) -> int:
    """Run the v4l2loopback playbook (and with ``--build``, rebuild)."""
    if sys.platform != "linux":
        sys.exit("v4l2:// and v4l2loopback are Linux-only")
    import ansible_runner

    cmdline = []
    passwords = {}
    password = become_password()
    if password is not None:
        cmdline.append("--ask-become-pass")
        passwords[r"^BECOME password.*:\s*?$"] = password
    if args.check:
        cmdline += ["--check", "--diff"]
    with tempfile.TemporaryDirectory(prefix="hilspy-setup-") as data:
        result = ansible_runner.run(
            private_data_dir=data,
            project_dir=str(PLAYBOOK.parent),
            playbook=PLAYBOOK.name,
            inventory={
                "all": {
                    "hosts": {
                        "localhost": {
                            "ansible_connection": "local",
                            # The system Python: it has the apt bindings, the venv does not.
                            "ansible_python_interpreter": "/usr/bin/python3",
                        }
                    }
                }
            },
            extravars=extravars(args),
            cmdline=" ".join(cmdline) or None,
            passwords=passwords,
            skip_tags=args.skip_tags,
            quiet=False,
        )
    if result.rc != 0 or args.check:
        return int(result.rc)
    # After the playbook: it installed the GStreamer development packages.
    if args.build and (rc := build()) != 0:
        return rc
    first = f"/dev/video{args.first}"
    print(f"\nready: hilspy.open('v4l2://{first}?width=640&height=480')")
    print("log out and in (or run `newgrp video`) if you were just added to `video`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
