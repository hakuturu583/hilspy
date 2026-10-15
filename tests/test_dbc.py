from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cantools
import pytest

from hilspy import dbc
from hilspy.can import Frame


@pytest.fixture
def db(vehicle_dbc: Path) -> dbc.Database:
    return dbc.load_file(vehicle_dbc)


@pytest.fixture
def reference(vehicle_dbc: Path) -> Any:
    return cantools.database.load_file(str(vehicle_dbc))


def test_metadata(db: dbc.Database) -> None:
    assert db.version == "1.0"
    assert db.nodes == ["VCU", "ADAS", "GNSS"]
    assert len(db) == 5 and "VehicleSpeed" in db and 0x100 in db and "Nope" not in db
    m = db.get_message("VehicleSpeed")
    assert (m.frame_id, m.length, m.cycle_time, m.senders) == (0x100, 8, 10, ["VCU"])
    assert m.comment == "Vehicle speed from the VCU"
    speed = m.get_signal("Speed")
    assert (speed.start, speed.length, speed.byte_order) == (0, 16, "little_endian")
    assert (speed.scale, speed.offset, speed.minimum, speed.maximum) == (0.01, 0, 0, 655.35)
    assert speed.unit == "km/h" and speed.receivers == ["ADAS"]
    assert speed.comment == "Vehicle speed over ground"
    gear = m.get_signal("Gear")
    assert gear.choices == {0: "P", 1: "R", 2: "N", 3: "D"} and gear.initial == 3
    diag = db.get_message(0x100000)
    assert diag.is_extended_frame and diag.name == "Diagnostic"
    mode, temp = diag.signals[0], diag.signals[1]
    assert mode.is_multiplexer and temp.multiplexer_signal == "Mode"
    assert temp.multiplexer_ranges == [(1, 1)]
    assert db.get_message("ImuData").get_signal("YawRate").is_float
    assert db.get_message("RadarObjects").is_fd
    with pytest.raises(dbc.MessageNotFoundError):
        db.get_message("Nope")
    with pytest.raises(dbc.MessageNotFoundError):
        m.get_signal("Nope")


def test_encode_decode(db: dbc.Database) -> None:
    values: dict[str, dbc.SignalValue] = {"Speed": 123.45, "Accel": -1.5, "Gear": "D", "Counter": 9}
    data = db.encode("VehicleSpeed", values)
    assert data == bytes([0x39, 0x30, 0xF4, 0x01, 0x03, 0x00, 0x00, 0x90])
    decoded = db.decode("VehicleSpeed", data)
    assert decoded["Gear"] == "D"
    assert decoded["Speed"] == pytest.approx(123.45)
    assert decoded["Accel"] == pytest.approx(-1.5)
    assert decoded["Counter"] == 9 and isinstance(decoded["Counter"], int)
    raw = db.decode(0x100, data, decode_choices=False, scaling=False)
    assert raw == {"Speed": 12345, "Accel": 500, "Gear": 3, "Counter": 9}
    assert db.encode("VehicleSpeed", raw, scaling=False) == data


def test_frames(db: dbc.Database) -> None:
    frame = db.encode_frame(
        "SteeringCmd", {"SteeringAngle": -90.5, "Torque": 1.25, "Enable": 1, "Checksum": 0xA5}
    )
    assert frame.arbitration_id == 0x200 and not frame.is_extended_id
    assert frame.data == bytes([0xFC, 0x77, 0x84, 0xD8, 0, 0, 0, 0xA5])
    decoded = db.decode_frame(frame)
    assert decoded["SteeringAngle"] == pytest.approx(-90.5)
    assert db.get_message("SteeringCmd").decode(frame) == decoded
    radar = db.encode_frame("RadarObjects", {"ObjectCount": 1, "Distance": 2.0, "Tail": 3})
    assert radar.is_fd and radar.bitrate_switch and len(radar.data) == 64
    with pytest.raises(dbc.MessageNotFoundError):
        db.decode_frame(Frame(0x7FF))


def test_strict_and_lenient(db: dbc.Database) -> None:
    with pytest.raises(dbc.EncodeError, match="missing value"):
        db.encode("VehicleSpeed", {"Speed": 1.0})
    with pytest.raises(dbc.EncodeError, match="maximum"):
        db.encode("VehicleSpeed", {"Speed": 700.0, "Accel": 0, "Gear": 0, "Counter": 0})
    with pytest.raises(dbc.EncodeError, match="not a value"):
        db.encode("VehicleSpeed", {"Speed": 1.0, "Accel": 0, "Gear": "X", "Counter": 0})
    with pytest.raises(dbc.EncodeError, match="no signal"):
        db.encode("VehicleSpeed", {"Typo": 1})
    with pytest.raises(TypeError):
        db.encode("VehicleSpeed", {"Speed": [1]})  # type: ignore[dict-item]
    data = db.encode("VehicleSpeed", {"Speed": 1.0}, strict=False)
    assert data == bytes([0x64, 0, 0, 0, 0x03, 0, 0, 0])  # Gear = GenSigStartValue
    with pytest.raises(dbc.EncodeError, match="does not fit"):
        db.encode("VehicleSpeed", {"Counter": 16}, strict=False)
    with pytest.raises(dbc.DecodeError):
        db.decode("VehicleSpeed", b"\x00")
    assert db.decode("VehicleSpeed", b"\x01\x00", allow_truncated=True) == {"Speed": 0.01}


def test_multiplexing(db: dbc.Database) -> None:
    data = db.encode("Diagnostic", {"Mode": "Power", "Voltage": 12.5, "ErrorCode": 7})
    assert db.decode("Diagnostic", data) == {"Mode": "Power", "Voltage": 12.5, "ErrorCode": 7}
    with pytest.raises(dbc.EncodeError, match="not present"):
        db.encode("Diagnostic", {"Mode": 2, "Temperature": 1.0})
    data = db.encode("Diagnostic", {"Mode": 1, "Temperature": -12.3})
    assert db.decode("Diagnostic", data) == pytest.approx({"Mode": "Thermal", "Temperature": -12.3})


def random_values(message: Any, rng: random.Random) -> dict[str, float]:
    """Raw values in range for every signal, as cantools-style physical values."""
    values: dict[str, float] = {}
    for s in message.signals:
        if s.is_float:
            raw: float = rng.uniform(-1e3, 1e3)
        elif s.is_signed:
            raw = rng.randint(-(1 << (s.length - 1)), (1 << (s.length - 1)) - 1)
        else:
            raw = rng.randint(0, (1 << s.length) - 1)
        values[s.name] = raw
    return values


@pytest.mark.parametrize("name", ["VehicleSpeed", "SteeringCmd", "ImuData", "RadarObjects"])
def test_matches_cantools(db: dbc.Database, reference: Any, name: str) -> None:
    """Raw payloads agree with cantools for random raw values, both directions."""
    rng = random.Random(name)
    ours = db.get_message(name)
    theirs = reference.get_message_by_name(name)
    for _ in range(200):
        raw = random_values(theirs, rng)
        expected = theirs.encode(raw, scaling=False, strict=False)
        assert ours.encode(raw, scaling=False, strict=False) == expected
        decoded = ours.decode(expected, scaling=False, decode_choices=False)
        assert decoded == pytest.approx(
            theirs.decode(expected, scaling=False, decode_choices=False)
        )


def test_multiplexed_matches_cantools(db: dbc.Database, reference: Any) -> None:
    theirs = reference.get_message_by_name("Diagnostic")
    for values in (
        {"Mode": 1, "Temperature": -100.5},
        {"Mode": 2, "Voltage": 30.123, "ErrorCode": 200},
    ):
        expected = theirs.encode(values)
        assert db.encode("Diagnostic", values) == expected
        assert db.decode("Diagnostic", expected, decode_choices=False) == pytest.approx(
            theirs.decode(expected, decode_choices=False)
        )


def test_parse_errors(tmp_path: Path) -> None:
    with pytest.raises(dbc.DbcError):
        dbc.load_string("not a dbc file")
    with pytest.raises(FileNotFoundError):
        dbc.load_file(tmp_path / "missing.dbc")


def test_windows_1252(tmp_path: Path, vehicle_dbc: Path) -> None:
    text = vehicle_dbc.read_text().replace('"degC"', '"°C"')
    path = tmp_path / "cp1252.dbc"
    path.write_bytes(text.encode("cp1252"))
    db = dbc.load_file(path)
    assert db.get_message("Diagnostic").get_signal("Temperature").unit == "°C"
