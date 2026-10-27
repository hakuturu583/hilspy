"""Channels over the in-process transport with the raw codec."""

from __future__ import annotations

import sys
import threading
import time

import pytest

import hilspy
from hilspy import Message


def test_round_trip(mem: str) -> None:
    with hilspy.open(mem) as a, hilspy.open(mem) as b:
        a.send(Message(data=b"\xaa", id=0x321))
        a.send(Message(data=b"bytes"))
        got = b.recv(timeout=1.0)
        assert got is not None and (got.name, got.id, got.data) == (None, 0x321, b"\xaa")
        assert got.timestamp is not None and abs(got.timestamp - time.time()) < 5
        got = b.recv(timeout=1.0)
        assert got is not None and got.data == b"bytes" and got.id is None
        assert a.recv(timeout=0) is None  # no echo by default


def test_recv_timeout(mem: str) -> None:
    with hilspy.open(mem) as ch:
        start = time.monotonic()
        assert ch.recv(timeout=0.1) is None
        assert 0.09 <= time.monotonic() - start < 1.0


def test_url_options(mem: str) -> None:
    with hilspy.open(f"{mem}?echo&filter=0x100:0x700") as ch:
        for i in (0x1AB, 0x2AB):
            ch.send(Message(data=b"", id=i))
        got = ch.recv(timeout=0)
        assert got is not None and got.id == 0x1AB
        assert ch.recv(timeout=0) is None


def test_iteration_ends_on_close(mem: str) -> None:
    rx = hilspy.open(mem)
    tx = hilspy.open(mem)
    received: list[Message] = []
    thread = threading.Thread(target=lambda: received.extend(rx))
    thread.start()
    for i in range(5):
        tx.send(Message(data=bytes([i])))
    deadline = time.monotonic() + 5
    while len(received) < 5 and time.monotonic() < deadline:
        time.sleep(0.01)
    rx.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert [m.data for m in received] == [bytes([i]) for i in range(5)]
    assert rx.closed
    with pytest.raises(hilspy.Error):
        rx.send(Message(data=b""))


def test_every_follows_updates(mem: str) -> None:
    # Driven by received messages, not sleeps: a shared CI runner can stall
    # the sender thread for tens of milliseconds.
    with hilspy.open(mem) as tx, hilspy.open(mem) as rx:
        task = tx.every(0.01, Message(data=b"\x01"))
        first = rx.recv(timeout=5.0)
        assert first is not None and first.data == b"\x01"
        task.update(Message(data=b"\x02"))
        task.period = 0.005
        assert task.period == 0.005
        payloads: list[bytes | None] = [first.data]
        deadline = time.monotonic() + 5.0
        while payloads[-1] != b"\x02":
            m = rx.recv(timeout=max(0.0, deadline - time.monotonic()))
            assert m is not None, "no message with the updated payload"
            payloads.append(m.data)
        task.stop()
        assert not task.running
        while (m := rx.recv(timeout=0)) is not None:
            payloads.append(m.data)
        assert len(payloads) == task.sent_count
        switch = payloads.index(b"\x02")
        assert set(payloads[:switch]) == {b"\x01"} and set(payloads[switch:]) == {b"\x02"}
        assert task.last_error is None
        task.stop()  # idempotent


def test_every_duration(mem: str) -> None:
    with hilspy.open(mem) as tx:
        with tx.every(0.01, Message(data=b""), duration=0.1) as task:
            deadline = time.monotonic() + 5
            while task.running and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not task.running
            assert 1 <= task.sent_count <= 11


def test_bad_urls_codecs_and_options(mem: str) -> None:
    with pytest.raises(hilspy.Error, match="unknown transport"):
        hilspy.open("pcan://0")
    with pytest.raises(hilspy.Error, match="scheme://"):
        hilspy.open("vcan0")
    with pytest.raises(hilspy.Error, match="unknown codec"):
        hilspy.open(mem, "nope")
    with pytest.raises(hilspy.Error, match="typo"):
        hilspy.open(f"{mem}?typo=1")
    with pytest.raises(hilspy.Error, match="typo"):
        hilspy.open(mem, "raw", typo=True)
    with hilspy.open(mem) as ch:
        with pytest.raises(hilspy.Error, match="raw codec"):
            ch.send("Named", {})
        with pytest.raises(TypeError):
            ch.send(Message(data=b""), {"x": 1})


def test_missing_can_interface() -> None:
    expected = OSError if sys.platform == "linux" else hilspy.Error
    with pytest.raises(expected):
        hilspy.open("can://hilspy-no-such-if0")


def test_file_record_and_replay(tmp_path: object, mem: str) -> None:
    import pathlib

    path = pathlib.Path(str(tmp_path)) / "rec.bin"
    with hilspy.open(f"file://{path}?write") as rec:
        rec.send(Message(data=b"abc"))
        rec.send(Message(data=b"def"))
    with hilspy.open(f"file://{path}") as replay:
        assert b"".join(m.data or b"" for m in replay) == b"abcdef"


def test_data_from_numpy_and_buffers() -> None:
    import array

    import numpy as np

    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    assert Message(data=image).data == image.tobytes()  # an image: pixels in C order
    view = image[:, ::2]  # not contiguous
    assert Message(data=view).data == view.tobytes()
    assert Message(data=memoryview(b"abc")).data == b"abc"
    assert Message(data=array.array("B", [1, 2])).data == b"\x01\x02"
    assert Message(data=bytearray(b"xy")).data == b"xy"
    assert Message(data=[1, 2, 255]).data == b"\x01\x02\xff"
    for wrong in (np.zeros(4, np.float32), np.zeros(2, np.uint16), array.array("h", [1])):
        with pytest.raises(TypeError, match="uint8"):
            Message(data=wrong)


def _drain(ch: hilspy.Channel) -> list[bytes]:
    out = []
    while (m := ch.recv(timeout=0.02)) is not None:
        assert m.data is not None
        out.append(m.data)
    return out


def test_disconnect_loses_what_the_link_carries(mem: str) -> None:
    with hilspy.open(mem) as a, hilspy.open(mem) as b:
        assert not a.disconnected()
        a.disconnect(direction="send")
        assert a.disconnected() and a.disconnected("send") and not a.disconnected("recv")
        a.send(Message(data=b"lost"))  # succeeds; the packet is lost
        b.send(Message(data=b"in"))
        assert _drain(b) == [] and _drain(a) == [b"in"]
        a.reconnect()
        a.send(Message(data=b"out"))
        assert _drain(b) == [b"out"]
        # The receive side: what arrives meanwhile stays lost after reconnecting.
        a.disconnect(direction="recv")
        b.send(Message(data=b"meanwhile"))
        time.sleep(0.01)
        a.reconnect(direction="recv")
        b.send(Message(data=b"after"))
        assert _drain(a) == [b"after"]
        with pytest.raises(hilspy.Error, match="send, recv or both"):
            a.disconnect(direction="tx")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            a.disconnect(-1.0)
        with pytest.raises(hilspy.Error, match="at least one"):
            a.disconnect(messages=[])
    with pytest.raises(hilspy.Error):
        a.disconnect()


def test_disconnect_for_a_while(mem: str) -> None:
    with hilspy.open(mem) as a, hilspy.open(mem) as b:
        a.disconnect(0.1)
        a.send(Message(data=b"lost"))
        assert _drain(b) == []
        time.sleep(0.12)
        assert not a.disconnected()
        a.send(Message(data=b"back"))
        assert _drain(b) == [b"back"]


def test_disconnect_stops_only_the_named_messages(mem: str, vehicle_dbc: str) -> None:
    with (
        hilspy.open(mem, vehicle_dbc, strict=False) as tx,
        hilspy.open(mem, vehicle_dbc) as rx,
    ):

        def received() -> list[str | None]:
            return [m.name for m in iter(lambda: rx.recv(timeout=0.02), None)]

        tx.disconnect(direction="send", messages=["VehicleSpeed"])
        tx.send("VehicleSpeed", {})
        tx.send("SteeringCmd", {})
        assert received() == ["SteeringCmd"]
        tx.reconnect()
        rx.disconnect(direction="recv", messages="SteeringCmd")
        tx.send("VehicleSpeed", {})
        tx.send("SteeringCmd", {})
        assert received() == ["VehicleSpeed"]
