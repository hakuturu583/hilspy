"""Prepare this Linux host for ``v4l2://``: GStreamer, v4l2loopback cameras.

Runs ``tools/ansible/v4l2loopback.yml`` on localhost through ansible-runner::

    uv run --group setup tools/setup_host.py                 # /dev/video10
    uv run --group setup tools/setup_host.py --devices 2 --build

It installs GStreamer and v4l2loopback (Debian / Ubuntu), creates the
devices now and at every boot, adds you to the ``video`` group and, with
``--build``, rebuilds this checkout with the ``v4l2`` feature. ``sudo`` is
used for the system changes; its password is asked for when needed.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PLAYBOOK = ROOT / "tools" / "ansible" / "v4l2loopback.yml"


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
    p.add_argument("--skip-tags", default="", help="ansible tags to skip: packages, module, build")
    args = p.parse_args(argv)
    if not 1 <= args.devices <= 8 or args.first < 0:
        p.error("--devices is 1..8 and --first a device number")
    return args


def extravars(args: argparse.Namespace) -> dict[str, Any]:
    uv = os.environ.get("UV") or shutil.which("uv")
    if args.build and not uv:
        sys.exit("--build needs uv (https://docs.astral.sh/uv/)")
    return {
        "v4l2loopback_count": args.devices,
        "v4l2loopback_first": args.first,
        "v4l2loopback_label": args.label,
        "v4l2loopback_exclusive_caps": not args.no_exclusive_caps,
        "hilspy_user": os.environ.get("SUDO_USER") or getpass.getuser(),
        "hilspy_build": args.build,
        "hilspy_source": str(ROOT),
        "hilspy_uv": uv or "uv",
    }


def become_password() -> str | None:
    """The sudo password, or None when root or sudo needs none."""
    if os.geteuid() == 0:
        return None
    ok = subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
    return None if ok else getpass.getpass("sudo password: ")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
            skip_tags=args.skip_tags or None,
            quiet=False,
        )
    if result.rc == 0 and not args.check:
        first = f"/dev/video{args.first}"
        print(f"\nready: hilspy.open('v4l2://{first}?width=640&height=480')")
        print("log out and in (or run `newgrp video`) if you were just added to `video`")
    return int(result.rc)


if __name__ == "__main__":
    sys.exit(main())
