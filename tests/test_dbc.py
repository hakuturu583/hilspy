"""The dbc codec, cross-checked against cantools."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cantools
import pytest

import hilspy
from hilspy import Message

DBC = Path(__file__).parent / "data" / "vehicle.dbc"


@pytest.fixture
def reference() -> Any:
    return cantools.database.load_file(str(DBC))


def test_send_and_receive(mem: str, vehicle_dbc: str) -> None:
    with (
        hilspy.open(mem, vehicle_dbc) as tx,
        hilspy.open(mem, vehicle_dbc) as rx,
        hilspy.open(mem) as raw,
    ):
        tx.send("VehicleSpeed", {"Speed": 123.45, "Accel": -1.5, "Gear": "D", "Counter": 9})
        m = rx.recv(timeout=1)
        assert m is not None and m.name == "VehicleSpeed" and m.id == 0x100
        assert m["Gear"] == "D" and m["Counter"] == 9 and isinstance(m["Counter"], int)
        assert m["Speed"] == pytest.approx(123.45) and m["Accel"] == pytest.approx(-1.5)
        r = raw.recv(timeout=1)
        assert r is not None and r.data == bytes([0x39, 0x30, 0xF4, 0x01, 0x03, 0, 0, 0x90])
        # Unknown identifiers come through as nameless messages.
        raw.send(Message(data=b"\x01", id=0x7FF))
        u = rx.recv(timeout=1)
        assert u is not None and (u.name, u.id) == (None, 0x7FF)


def test_big_endian_and_fd(mem: str, vehicle_dbc: str) -> None:
    with hilspy.open(mem, vehicle_dbc) as tx, hilspy.open(mem) as raw:
        tx.send(
            "SteeringCmd",
            {"SteeringAngle": -90.5, "Torque": 1.25, "Enable": 1, "Checksum": 0xA5},
        )
        tx.send("RadarObjects", {"ObjectCount": 1, "Distance": 2.0, "Tail": 3})
        steer, radar = raw.recv(timeout=1), raw.recv(timeout=1)
        assert steer is not None and steer.data == bytes([0xFC, 0x77, 0x84, 0xD8, 0, 0, 0, 0xA5])
        assert radar is not None and radar.fd and len(radar.data or b"") == 64


def test_strict_and_options(mem: str, vehicle_dbc: str) -> None:
    with hilspy.open(mem, vehicle_dbc) as tx, hilspy.open(mem) as raw:
        with pytest.raises(hilspy.Error, match="missing value"):
            tx.send("VehicleSpeed", {"Speed": 1.0})
        with pytest.raises(hilspy.Error, match="maximum"):
            tx.send("VehicleSpeed", {"Speed": 700.0, "Accel": 0, "Gear": 0, "Counter": 0})
        with pytest.raises(hilspy.Error, match="not a value"):
            tx.send("VehicleSpeed", {"Speed": 1.0, "Accel": 0, "Gear": "X", "Counter": 0})
        with pytest.raises(hilspy.Error, match="no signal"):
            tx.send("VehicleSpeed", {"Typo": 1})
        with pytest.raises(hilspy.Error, match="no message"):
            tx.send("Nope", {})
    with hilspy.open(mem, vehicle_dbc, strict=False) as tx, hilspy.open(mem) as raw:
        tx.send("VehicleSpeed", {"Speed": 1.0})
        r = raw.recv(timeout=1)
        assert r is not None and r.data == bytes([0x64, 0, 0, 0, 0x03, 0, 0, 0])  # Gear initial
    with pytest.raises(hilspy.Error, match="needs a file"):
        hilspy.open(mem, "dbc")
    with pytest.raises(FileNotFoundError):
        hilspy.open(mem, "dbc:/nonexistent.dbc")


def test_multiplexing(mem: str, vehicle_dbc: str) -> None:
    with hilspy.open(mem, vehicle_dbc) as tx, hilspy.open(mem, vehicle_dbc) as rx:
        tx.send("Diagnostic", {"Mode": "Power", "Voltage": 12.5, "ErrorCode": 7})
        m = rx.recv(timeout=1)
        assert m is not None and m.fields == {"Mode": "Power", "Voltage": 12.5, "ErrorCode": 7}
        with pytest.raises(hilspy.Error, match="not present"):
            tx.send("Diagnostic", {"Mode": 2, "Temperature": 1.0})


def random_raw(message: Any, rng: random.Random) -> dict[str, float]:
    values: dict[str, float] = {}
    for s in message.signals:
        if s.is_float:
            values[s.name] = rng.uniform(-1e3, 1e3)
        elif s.is_signed:
            values[s.name] = rng.randint(-(1 << (s.length - 1)), (1 << (s.length - 1)) - 1)
        else:
            values[s.name] = rng.randint(0, (1 << s.length) - 1)
    return values


@pytest.mark.parametrize("name", ["VehicleSpeed", "SteeringCmd", "ImuData", "RadarObjects"])
def test_matches_cantools(mem: str, reference: Any, name: str) -> None:
    """Payloads agree with cantools for random raw values, both directions."""
    rng = random.Random(name)
    theirs = reference.get_message_by_name(name)
    raw_values: dict[str, Any] = {"scaling": False, "strict": False, "decode_choices": False}
    spec = f"dbc:{DBC}"
    encode_bus, decode_bus = f"{mem}-encode", f"{mem}-decode"
    with (
        hilspy.open(encode_bus, spec, **raw_values) as encoder,
        hilspy.open(encode_bus) as encoded,
        hilspy.open(decode_bus) as to_decode,
        hilspy.open(decode_bus, spec, **raw_values) as decoder,
    ):
        for _ in range(100):
            values = random_raw(theirs, rng)
            expected = theirs.encode(values, scaling=False, strict=False)
            encoder.send(name, values)
            sent = encoded.recv(timeout=1)
            assert sent is not None and sent.data == expected
            to_decode.send(Message(data=expected, id=theirs.frame_id, fd=len(expected) > 8))
            got = decoder.recv(timeout=1)
            assert got is not None and got.name == name
            decoded = theirs.decode(expected, scaling=False, decode_choices=False)
            assert got.fields == pytest.approx(decoded)


def test_windows_1252(tmp_path: Path, mem: str) -> None:
    text = DBC.read_text().replace('"degC"', '"°C"')
    path = tmp_path / "cp1252.dbc"
    path.write_bytes(text.encode("cp1252"))
    with hilspy.open(mem, f"dbc:{path}") as ch:
        assert not ch.closed
