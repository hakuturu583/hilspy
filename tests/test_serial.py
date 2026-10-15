"""serial:// over a pseudo-terminal (Linux)."""

from __future__ import annotations

import os
import sys

import pytest

import hilspy

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="uses a Linux pseudo-terminal")


def test_sbf_over_a_serial_port() -> None:
    import tty  # Unix only: importing it at module level breaks collection on Windows

    master, slave = os.openpty()
    tty.setraw(master)
    path = os.ttyname(slave)
    try:
        with hilspy.open(f"serial://{path}?baud=115200", "sbf") as receiver:
            receiver.send("AttEuler", {"heading": 45.0})
            data = os.read(master, 1024)
            assert data.startswith(b"$@")
            # The device answers with the same block, split across writes.
            os.write(master, data[:5])
            os.write(master, data[5:])
            msg = receiver.recv(timeout=2)
            assert msg is not None and msg.name == "AttEuler" and msg["heading"] == 45.0
    finally:
        os.close(master)
        os.close(slave)


def test_bad_port() -> None:
    with pytest.raises(hilspy.Error, match="cannot open"):
        hilspy.open("serial:///nonexistent/ttyX")
    with pytest.raises(hilspy.Error, match="parity"):
        hilspy.open("serial:///dev/null?parity=mark")
