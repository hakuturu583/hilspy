"""can:// on a real vcan interface.

They run when ``HILSPY_TEST_VCAN`` names one, e.g.::

    sudo modprobe vcan
    sudo ip link add dev vcan0 type vcan && sudo ip link set vcan0 mtu 72 up
    HILSPY_TEST_VCAN=vcan0 uv run pytest tests/test_socketcan.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import hilspy
from hilspy import Message

VCAN = os.environ.get("HILSPY_TEST_VCAN")
pytestmark = pytest.mark.skipif(not VCAN, reason="HILSPY_TEST_VCAN is not set")
DBC = f"dbc:{Path(__file__).parent / 'data' / 'vehicle.dbc'}"


def drain(ch: hilspy.Channel) -> None:
    while ch.recv(timeout=0.01) is not None:
        pass


def test_raw_round_trip() -> None:
    with hilspy.open(f"can://{VCAN}") as a, hilspy.open(f"can://{VCAN}") as b:
        drain(b)
        a.send(Message(data=b"\x01\x02\x03", id=0x123))
        a.send(Message(data=b"\xff", id=0x1ABCDE))
        m1, m2 = b.recv(timeout=1), b.recv(timeout=1)
        assert m1 is not None and (m1.id, m1.data) == (0x123, b"\x01\x02\x03")
        assert m2 is not None and m2.extended and m2.id == 0x1ABCDE
        assert a.recv(timeout=0.05) is None
        with pytest.raises(hilspy.Error):
            a.send(Message(data=b"no id"))


def test_dbc_and_fd() -> None:
    with hilspy.open(f"can://{VCAN}?fd", DBC) as a, hilspy.open(f"can://{VCAN}?fd", DBC) as b:
        drain(b)
        a.send("RadarObjects", {"ObjectCount": 2, "Distance": 1.0, "Tail": 7})
        m = b.recv(timeout=1)
        assert m is not None and m.fd and m.name == "RadarObjects" and m["Tail"] == 7
    with hilspy.open(f"can://{VCAN}") as classic:
        with pytest.raises(hilspy.Error, match="fd"):
            classic.send(Message(data=bytes(16), id=1))


def test_kernel_filters_and_echo() -> None:
    with hilspy.open(f"can://{VCAN}?echo&filter=0x200:0x700") as a:
        drain(a)
        a.send(Message(data=b"", id=0x2AA))
        a.send(Message(data=b"", id=0x3AA))
        a.send(Message(data=b"", id=0x2AA, extended=True))
        m = a.recv(timeout=1)
        assert m is not None and (m.id, m.extended) == (0x2AA, False)
        assert a.recv(timeout=0.05) is None


def test_periodic() -> None:
    with hilspy.open(f"can://{VCAN}") as tx, hilspy.open(f"can://{VCAN}") as rx:
        drain(rx)
        with tx.every(0.01, Message(data=b"\x00", id=0x42), duration=0.1) as task:
            frames = []
            while (m := rx.recv(timeout=0.3)) is not None:
                frames.append(m)
            assert len(frames) == task.sent_count >= 1
