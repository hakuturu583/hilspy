"""AUTOSAR E2E protection of DBC messages (the dbc codec's e2e option)."""

from __future__ import annotations

from pathlib import Path

import pytest

import hilspy

DBC = f"dbc:{Path(__file__).parent / 'data' / 'e2e.dbc'}"
E2E = {
    "BrakeCmd": {"profile": 11, "data_id": 0x123, "signals": ["Crc", "Counter", "DataIdNibble"]},
    "SafetyStatus": {"profile": 5, "data_id": 0x4242, "signals": ["Crc", "Counter"]},
}


def drain(ch: hilspy.Channel) -> list[hilspy.Message]:
    out = []
    while (m := ch.recv(timeout=0.05)) is not None:
        out.append(m)
    return out


def test_periodic_protected_messages(mem: str) -> None:
    clock = hilspy.Clock()
    with (
        hilspy.open(mem, DBC, e2e=E2E, clock=clock) as ecu,
        hilspy.open(mem, DBC, e2e=E2E) as monitor,
        hilspy.open(mem) as raw,
    ):
        brake = ecu.every(0.01, "BrakeCmd", {"Pressure": 12.5, "Request": 1})
        clock.advance(0.2)
        got = [m for m in drain(monitor) if m.name == "BrakeCmd"]
        assert brake.sent_count == 21 and len(got) == 21
        assert [m["Counter"] for m in got[:16]] == [*range(15), 0]
        assert all(m["e2e"] == "Ok" and m["Pressure"] == 12.5 for m in got)
        # A tampered frame fails the check.
        assert len(drain(raw)) == 21
        last = got[-1]
        tampered = bytearray(last.data or b"")
        tampered[2] ^= 0xFF
        raw.send(hilspy.Message(data=bytes(tampered), id=last.id))
        (bad,) = drain(monitor)
        assert bad["e2e"] == "CrcError"


def test_bad_config(mem: str) -> None:
    with pytest.raises(hilspy.Error, match="unsupported profile"):
        hilspy.open(mem, DBC, e2e={"BrakeCmd": {"profile": 3}})
    with pytest.raises(hilspy.Error, match="Nope"):
        hilspy.open(mem, DBC, e2e={"Nope": {"profile": 11}})
