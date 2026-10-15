"""LIN and FlexRay frames: Message headers, mem:// and the fibex codec."""

from __future__ import annotations

import pytest

import hilspy
from hilspy import Message


def test_lin_message() -> None:
    m = Message(data=bytes([0x4A, 0x55, 0x93, 0xE5]), id=0x10, bus="lin")
    assert (m.bus, m.id, m.enhanced, m.extended, m.fd) == ("lin", 0x10, True, False, False)
    # Enhanced: over the protected id 0x50 and the data (hand-computed).
    assert m.checksum == 0xFF - 0x69
    classic = Message(data=bytes([0x4A, 0x55, 0x93, 0xE5]), id=0x10, enhanced=False)
    assert (classic.bus, classic.enhanced, classic.checksum) == ("lin", False, 0xFF - 0x19)
    # Diagnostic frames default to (and always compute) classic.
    diag = Message(data=bytes(8), id=0x3C, bus="lin")
    assert not diag.enhanced and diag.checksum == 0xFF
    assert Message(data=b"\x01", id=1, checksum=0x12).checksum == 0x12
    assert m.cycle is None and m.channel is None
    assert 'bus="lin"' in repr(m)


def test_flexray_message() -> None:
    m = Message(data=bytes(4), id=5, bus="flexray", cycle=3, channel="B")
    assert (m.bus, m.id, m.cycle, m.channel) == ("flexray", 5, 3, "B")
    assert m.checksum is None and not m.enhanced
    d = Message(data=bytes(2), id=7, cycle=0)
    assert (d.bus, d.channel) == ("flexray", "A")
    assert Message(data=bytes(2), id=7, channel="ab").channel == "AB"
    can = Message(data=b"", id=0x123)
    assert (can.bus, can.cycle, can.checksum) == ("can", None, None)
    assert Message(data=b"x").bus is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": 64, "bus": "lin"},
        {"id": 1, "bus": "lin", "data": bytes(9)},
        {"id": 1, "bus": "lin", "fd": True},
        {"id": 1, "enhanced": True, "cycle": 1},
        {"id": 0, "bus": "flexray"},
        {"id": 2048, "bus": "flexray"},
        {"id": 1, "cycle": 64},
        {"id": 1, "cycle": 0, "data": b"odd"},
        {"id": 1, "channel": "C"},
        {"id": 1, "bus": "most"},
        {"bus": "lin", "data": b""},
        {"id": 0x123, "cycle": 1, "bus": "can"},
    ],
)
def test_header_validation(kwargs: dict[str, object]) -> None:
    kwargs.setdefault("data", b"")
    with pytest.raises(ValueError):
        Message(**kwargs)  # type: ignore[arg-type]


def test_headers_travel_over_mem(mem: str) -> None:
    with hilspy.open(mem) as a, hilspy.open(mem) as b:
        a.send(Message(data=b"\x01\x02", id=0x21, bus="lin", checksum=0x00))
        a.send(Message(data=b"\x01\x02", id=12, cycle=33, channel="AB"))
        lin = b.recv(timeout=1)
        assert lin is not None
        assert (lin.bus, lin.id, lin.enhanced, lin.checksum) == ("lin", 0x21, True, 0)
        fr = b.recv(timeout=1)
        assert fr is not None
        assert (fr.bus, fr.id, fr.cycle, fr.channel) == ("flexray", 12, 33, "AB")
        assert fr.data == b"\x01\x02"
