"""SocketCAN tests on a real vcan interface.

They run when ``HILSPY_TEST_VCAN`` names one, e.g.::

    sudo modprobe vcan
    sudo ip link add dev vcan0 type vcan && sudo ip link set vcan0 mtu 72 up
    HILSPY_TEST_VCAN=vcan0 uv run pytest tests/test_socketcan.py
"""

from __future__ import annotations

import os

import pytest

from hilspy.can import Bus, Filter, Frame

VCAN = os.environ.get("HILSPY_TEST_VCAN")
pytestmark = pytest.mark.skipif(not VCAN, reason="HILSPY_TEST_VCAN is not set")


def drain(bus: Bus) -> None:
    while bus.recv(timeout=0.01) is not None:
        pass


def test_classic_round_trip() -> None:
    assert VCAN
    with Bus(VCAN) as a, Bus(VCAN) as b:
        drain(b)
        a.send(Frame(0x123, b"\x01\x02\x03"))
        a.send(Frame(0x1ABCDE, b"\xff", is_extended_id=True))
        a.send(Frame(0x7, is_remote_frame=True, dlc=3))
        f1, f2, f3 = (b.recv(timeout=1.0) for _ in range(3))
        assert f1 is not None and (f1.arbitration_id, f1.data) == (0x123, b"\x01\x02\x03")
        assert f2 is not None and f2.is_extended_id and f2.arbitration_id == 0x1ABCDE
        assert f3 is not None and f3.is_remote_frame and f3.dlc == 3
        assert a.recv(timeout=0.05) is None


def test_fd_round_trip() -> None:
    assert VCAN
    with Bus(VCAN, fd=True) as a, Bus(VCAN, fd=True) as b:
        drain(b)
        a.send(Frame(0x10, bytes(range(64)), is_fd=True, bitrate_switch=True))
        got = b.recv(timeout=1.0)
        assert got is not None and got.is_fd and got.bitrate_switch
        assert got.data == bytes(range(64))


def test_kernel_filters_and_own_messages() -> None:
    assert VCAN
    with Bus(VCAN, receive_own_messages=True, filters=[Filter(0x200, 0x700, extended=False)]) as a:
        drain(a)
        a.send(Frame(0x2AA))
        a.send(Frame(0x3AA))
        a.send(Frame(0x2AA, is_extended_id=True))
        got = a.recv(timeout=1.0)
        assert got is not None and (got.arbitration_id, got.is_extended_id) == (0x2AA, False)
        assert a.recv(timeout=0.05) is None


def test_periodic() -> None:
    assert VCAN
    with Bus(VCAN) as tx, Bus(VCAN) as rx:
        drain(rx)
        with tx.send_periodic(Frame(0x42, b"\x00"), period=0.01, duration=0.1) as task:
            frames = []
            while (f := rx.recv(timeout=0.2)) is not None:
                frames.append(f)
            assert len(frames) == task.sent_count >= 8
