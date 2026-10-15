"""The public surface and Message."""

from __future__ import annotations

import pytest

import hilspy
from hilspy import Message


def test_public_surface_is_minimal() -> None:
    public = sorted(n for n in dir(hilspy) if not n.startswith("_"))
    assert public == [
        "Channel",
        "Error",
        "Message",
        "Periodic",
        "merge",
        "open",
        "register_codec",
        "register_transport",
    ]


def test_named_message() -> None:
    m = Message("VehicleSpeed", {"Speed": 12.5, "Gear": "D", "On": True, "Sub": {"a": 1}})
    assert m.name == "VehicleSpeed"
    assert m.fields == {"Speed": 12.5, "Gear": "D", "On": True, "Sub": {"a": 1}}
    assert m["Speed"] == 12.5 and "Gear" in m and "Nope" not in m
    with pytest.raises(KeyError):
        m["Nope"]
    assert m.data is None and m.id is None and m.timestamp is None
    assert m == Message("VehicleSpeed", {"Speed": 12.5, "Gear": "D", "On": True, "Sub": {"a": 1}})


def test_raw_message() -> None:
    m = Message(data=b"\x01\x02", id=0x123)
    assert (m.name, m.data, m.id, m.extended, m.fd) == (None, b"\x01\x02", 0x123, False, False)
    assert Message(data=[1], id=0x800).extended
    assert Message(data=bytes(12), id=1, fd=True).fd
    assert Message(data=bytearray(b"x")).id is None
    assert "0x123" in repr(m)


def test_message_validation() -> None:
    with pytest.raises(ValueError):
        Message()
    with pytest.raises(ValueError):
        Message(data=b"", fd=True)
    with pytest.raises(TypeError):
        Message(data="text")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Message("x", {"v": object()})  # type: ignore[dict-item]
