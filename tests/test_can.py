from __future__ import annotations

import sys
import threading
import time

import pytest

from hilspy.can import (
    Bus,
    BusClosedError,
    Filter,
    Frame,
    InvalidFrameError,
    UnsupportedInterfaceError,
)


def test_frame_fields() -> None:
    f = Frame(0x123, b"\x01\x02")
    assert (f.arbitration_id, f.data, f.dlc, len(f)) == (0x123, b"\x01\x02", 2, 2)
    assert not f.is_extended_id and not f.is_fd and not f.is_remote_frame
    assert f.timestamp is None
    assert Frame(0x123, [1, 2]) == f
    assert Frame(0x123, bytearray(b"\x01\x02")) == f
    assert Frame(0x800).is_extended_id
    assert Frame(0x10, is_extended_id=True).is_extended_id
    r = Frame(0x10, is_remote_frame=True, dlc=4)
    assert (r.is_remote_frame, r.dlc, r.data) == (True, 4, b"")
    fd = Frame(0x10, bytes(64), is_fd=True, bitrate_switch=True)
    assert fd.is_fd and fd.bitrate_switch and fd.dlc == 64
    assert "0x123" in repr(f)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arbitration_id": 0x800, "is_extended_id": False},
        {"arbitration_id": 0x2000_0000},
        {"arbitration_id": 1, "data": bytes(9)},
        {"arbitration_id": 1, "data": bytes(13), "is_fd": True},
    ],
)
def test_invalid_frames(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidFrameError):
        Frame(**kwargs)  # type: ignore[arg-type]


def test_frame_argument_errors() -> None:
    with pytest.raises(ValueError):
        Frame(1, b"\x00", dlc=2)
    with pytest.raises(ValueError):
        Frame(1, b"\x00", bitrate_switch=True)
    with pytest.raises(TypeError):
        Frame(1, "text")  # type: ignore[arg-type]
    with pytest.raises((TypeError, OverflowError)):
        Frame(1, [256])


def test_virtual_bus_round_trip(channel: str) -> None:
    with Bus(channel, "virtual") as a, Bus(channel, "virtual") as b:
        a.send(Frame(0x321, b"\xaa"))
        got = b.recv(timeout=1.0)
        assert got is not None
        assert (got.arbitration_id, got.data) == (0x321, b"\xaa")
        assert got.timestamp is not None and abs(got.timestamp - time.time()) < 5
        # The sender does not receive its own frame unless asked to.
        assert a.recv(timeout=0) is None


def test_recv_timeout(channel: str) -> None:
    with Bus(channel, "virtual") as bus:
        start = time.monotonic()
        assert bus.recv(timeout=0.1) is None
        assert 0.09 <= time.monotonic() - start < 1.0


def test_receive_own_messages_and_filters(channel: str) -> None:
    with Bus(
        channel,
        "virtual",
        receive_own_messages=True,
        filters=[Filter(0x100, 0x700, extended=False), {"can_id": 0x1234, "extended": True}],
    ) as bus:
        for frame in (
            Frame(0x1AB),
            Frame(0x2AB),
            Frame(0x1AB, is_extended_id=True),
            Frame(0x1234),
        ):
            bus.send(frame)
        ids = []
        while (f := bus.recv(timeout=0)) is not None:
            ids.append((f.arbitration_id, f.is_extended_id))
        assert ids == [(0x1AB, False), (0x1234, True)]
        bus.set_filters(None)
        bus.send(Frame(0x2AB))
        received = bus.recv(timeout=0)
        assert received is not None and received.arbitration_id == 0x2AB


def test_filter_matches() -> None:
    f = Filter(0x100, 0x7F0)
    assert f.matches(Frame(0x10F)) and not f.matches(Frame(0x110))
    assert Filter(0x10).can_mask == 0x1FFF_FFFF
    assert Filter(0x10, extended=False).can_mask == 0x7FF


def test_fd_requires_fd_bus(channel: str) -> None:
    with Bus(channel, "virtual") as classic, Bus(channel, "virtual", fd=True) as fd:
        frame = Frame(0x10, bytes(16), is_fd=True)
        with pytest.raises(InvalidFrameError):
            classic.send(frame)
        fd.send(frame)
        fd.send(Frame(0x11, b"\x01"))
        got = classic.recv(timeout=0)
        assert got is not None and got.arbitration_id == 0x11  # FD frame not seen


def test_iteration_ends_on_shutdown(channel: str) -> None:
    bus = Bus(channel, "virtual")
    sender = Bus(channel, "virtual")
    received: list[Frame] = []

    def consume() -> None:
        received.extend(bus)

    thread = threading.Thread(target=consume)
    thread.start()
    for i in range(5):
        sender.send(Frame(i))
    deadline = time.monotonic() + 2
    while len(received) < 5 and time.monotonic() < deadline:
        time.sleep(0.01)
    bus.shutdown()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert [f.arbitration_id for f in received] == [0, 1, 2, 3, 4]
    assert bus.closed
    with pytest.raises(BusClosedError):
        bus.send(Frame(1))


def test_periodic_task(channel: str) -> None:
    with Bus(channel, "virtual") as tx, Bus(channel, "virtual") as rx:
        task = tx.send_periodic(Frame(0x50, b"\x01"), period=0.01)
        time.sleep(0.105)
        task.modify(Frame(0x50, b"\x02"))
        task.period = 0.005
        time.sleep(0.05)
        task.stop()
        assert not task.is_running
        sent = task.sent_count
        # Loose bounds: timers are coarse (~15 ms on Windows) and shared CI
        # runners can stall the sender thread.
        assert 2 <= sent <= 40
        frames = []
        while (f := rx.recv(timeout=0)) is not None:
            frames.append(f.data)
        assert len(frames) == sent
        assert frames[0] == b"\x01" and frames[-1] == b"\x02"
        assert task.last_error is None
        task.stop()  # idempotent


def test_periodic_task_duration(channel: str) -> None:
    with Bus(channel, "virtual") as tx:
        with tx.send_periodic(Frame(0x50), period=0.01, duration=0.1) as task:
            deadline = time.monotonic() + 5
            while task.is_running and time.monotonic() < deadline:
                time.sleep(0.01)
            # The task ends on its own. Counts only get an upper bound: a
            # shared CI runner can stall the sender thread for tens of ms.
            assert not task.is_running
            assert 1 <= task.sent_count <= 11
            with pytest.raises(BusClosedError):
                task.modify(Frame(0x51))


def test_unknown_interface() -> None:
    with pytest.raises(UnsupportedInterfaceError):
        Bus("x", "pcan")


def test_missing_socketcan_interface() -> None:
    expected = OSError if sys.platform == "linux" else UnsupportedInterfaceError
    with pytest.raises(expected):
        Bus("hilspy-no-such-if0", "socketcan")
