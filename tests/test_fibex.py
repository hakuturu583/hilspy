"""The fibex codec, cross-checked against the equivalent DBC through the dbc codec."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cantools
import pytest

import hilspy
from hilspy import Message

DATA = Path(__file__).parent / "data"
FIBEX = f"fibex:{DATA / 'network.xml'}"
DBC = f"dbc:{DATA / 'network.dbc'}"
BODY: dict[str, Any] = {"channel": "Body"}

# Physical values for every frame of the Body channel.
VALUES: dict[str, dict[str, Any]] = {
    "VehicleStatus": {"Speed": 123.45, "Gear": "D", "BrakePressure": 12.5, "BrakeActive": 1},
    "SteeringCmd": {"SteeringAngle": -90.5, "Torque": "Invalid", "Enable": 1},
    "RadarObjects": {"ObjectCount": 3, "Distance": 42.0, "RadarStatus": "Degraded"},
}


def drain(ch: hilspy.Channel) -> list[Message]:
    out = []
    while (m := ch.recv(timeout=0.05)) is not None:
        out.append(m)
    return out


def test_same_payloads_as_the_dbc(mem: str) -> None:
    fibex_bus, dbc_bus = f"{mem}-fibex", f"{mem}-dbc"
    with (
        hilspy.open(fibex_bus, FIBEX, **BODY) as fibex,
        hilspy.open(fibex_bus) as fibex_raw,
        hilspy.open(dbc_bus, DBC) as dbc,
        hilspy.open(dbc_bus) as dbc_raw,
    ):
        for name, values in VALUES.items():
            fibex.send(name, values)
            dbc.send(name, values)
            a, b = fibex_raw.recv(timeout=1), dbc_raw.recv(timeout=1)
            assert a is not None and b is not None
            assert (a.data, a.id, a.extended, a.fd) == (b.data, b.id, b.extended, b.fd), name
    with hilspy.open(mem, FIBEX, **BODY) as tx, hilspy.open(mem) as raw:
        tx.send("SteeringCmd", VALUES["SteeringCmd"])
        steer = raw.recv(timeout=1)
        assert steer is not None and steer.extended and steer.id == 0x18FF1234
        assert steer.data == bytes([0xFC, 0x77, 0x7F, 0x01, 0, 0, 0, 0])
        tx.send("RadarObjects", VALUES["RadarObjects"])
        radar = raw.recv(timeout=1)
        assert radar is not None and radar.fd and len(radar.data or b"") == 16


@pytest.mark.parametrize(("sender", "receiver"), [(FIBEX, DBC), (DBC, FIBEX)])
def test_decodes_the_others_frames(mem: str, sender: str, receiver: str) -> None:
    with (
        hilspy.open(mem, sender, **(BODY if sender == FIBEX else {})) as tx,
        hilspy.open(mem, receiver, **(BODY if receiver == FIBEX else {})) as rx,
    ):
        for name, values in VALUES.items():
            tx.send(name, values)
            m = rx.recv(timeout=1)
            assert m is not None and m.name == name
            for k, v in values.items():
                assert m[k] == (pytest.approx(v) if isinstance(v, float) else v)


def random_raw(message: Any, rng: random.Random) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for s in message.signals:
        if s.is_signed:
            values[s.name] = rng.randint(-(1 << (s.length - 1)), (1 << (s.length - 1)) - 1)
        else:
            values[s.name] = rng.randint(0, (1 << s.length) - 1)
    return values


@pytest.mark.parametrize("name", list(VALUES))
def test_random_raw_values_match(mem: str, name: str) -> None:
    rng = random.Random(name)
    reference: Any = cantools.database.load_file(str(DATA / "network.dbc"))
    theirs = reference.get_message_by_name(name)
    raw: dict[str, Any] = {"scaling": False, "strict": False, "decode_choices": False}
    fibex_bus, dbc_bus = f"{mem}-fibex", f"{mem}-dbc"
    with (
        hilspy.open(fibex_bus, FIBEX, **BODY, **raw) as fibex,
        hilspy.open(fibex_bus) as fibex_raw,
        hilspy.open(dbc_bus, DBC, **raw) as dbc,
        hilspy.open(dbc_bus) as dbc_raw,
    ):
        for _ in range(50):
            values = random_raw(theirs, rng)
            fibex.send(name, values)
            dbc.send(name, values)
            a, b = fibex_raw.recv(timeout=1), dbc_raw.recv(timeout=1)
            assert a is not None and b is not None and a.data == b.data
            # And the bytes decode back to the same values through either codec.
            fibex_raw.send(b)
            back = fibex.recv(timeout=1)
            assert back is not None and back.fields == values


def test_channels_and_qualified_names(mem: str) -> None:
    with pytest.raises(hilspy.Error, match=r"Body \(CH_BODY\), Chassis \(CH_CHASSIS\)"):
        hilspy.open(mem, FIBEX)
    with pytest.raises(hilspy.Error, match="Powertrain"):
        hilspy.open(mem, FIBEX, channel="Powertrain")
    with (
        hilspy.open(mem, FIBEX, channel="CH_CHASSIS") as tx,
        hilspy.open(mem, FIBEX, channel="Chassis") as rx,
    ):
        wheels = {"Front.Left": 1.0, "Front.Right": 2.0, "Rear.Left": 3.0, "Rear.Right": 4.0}
        tx.send("WheelSpeeds", wheels)
        m = rx.recv(timeout=1)
        assert m is not None and (m.name, m.id) == ("WheelSpeeds", 0x300)
        assert m.fields == wheels
        # The Body channel's identifiers mean nothing here.
        tx.send(Message(data=bytes(8), id=0x100))
        u = rx.recv(timeout=1)
        assert u is not None and (u.name, u.id) == (None, 0x100)


def test_options(mem: str) -> None:
    with hilspy.open(mem, FIBEX, **BODY) as tx:
        with pytest.raises(hilspy.Error, match="missing value"):
            tx.send("VehicleStatus", {"Speed": 1.0})
        with pytest.raises(hilspy.Error, match="maximum"):
            tx.send("VehicleStatus", {**VALUES["VehicleStatus"], "Speed": 700.0})
    with pytest.raises(hilspy.Error, match="does not take"):
        hilspy.open(mem, FIBEX, channel="Body", typo=1)
    with pytest.raises(hilspy.Error, match="needs a file"):
        hilspy.open(mem, "fibex")
    with pytest.raises(FileNotFoundError):
        hilspy.open(mem, "fibex:/nonexistent.xml")


def test_every_on_a_clock(mem: str) -> None:
    clock = hilspy.Clock(start=5.0)
    with (
        hilspy.open(mem, FIBEX, clock=clock, **BODY) as ecu,
        hilspy.open(mem, DBC) as rx,
    ):
        status = ecu.every(0.01, "VehicleStatus", VALUES["VehicleStatus"])
        clock.advance(0.05)
        assert status.sent_count == 6
        got = drain(rx)
        assert [m.name for m in got] == ["VehicleStatus"] * 6
        assert all(m["Gear"] == "D" for m in got)
        status.update("VehicleStatus", {**VALUES["VehicleStatus"], "Gear": "R"})
        clock.advance(0.01)
        got = drain(rx)
        assert [m["Gear"] for m in got] == ["R"]


def test_every_without_a_period_uses_the_cyclic_timing(mem: str) -> None:
    clock = hilspy.Clock(start=0.0)
    with (
        hilspy.open(mem, FIBEX, clock=clock, **BODY) as ecu,
        hilspy.open(mem, DBC) as rx,
    ):
        # CYCLIC-TIMING: VehicleStatus PT0.01S, RadarObjects PT0.05S.
        status = ecu.every(None, "VehicleStatus", VALUES["VehicleStatus"])
        radar = ecu.every(None, "RadarObjects", VALUES["RadarObjects"])
        assert (status.period, radar.period) == pytest.approx((0.01, 0.05))
        clock.advance(0.1)
        assert (status.sent_count, radar.sent_count) == (11, 3)
        names = [m.name for m in drain(rx)]
        assert (names.count("VehicleStatus"), names.count("RadarObjects")) == (11, 3)
        with pytest.raises(hilspy.Error, match='"SteeringCmd" has no cycle time'):
            ecu.every(None, "SteeringCmd", VALUES["SteeringCmd"])
    # The timing is the Body channel's triggering's, not the frame's.
    with (
        hilspy.open(mem, FIBEX, channel="Chassis") as chassis,
        pytest.raises(hilspy.Error, match='"VehicleStatus" has no cycle time'),
    ):
        chassis.every(None, "VehicleStatus", VALUES["VehicleStatus"])


def test_acts_as_an_ecu(mem: str) -> None:
    # Body: VehicleStatus Gateway -> Adas, SteeringCmd Adas -> Gateway,
    # RadarObjects Radar -> Adas (the ECUs' connector ports).
    with (
        hilspy.open(mem, FIBEX, ecu="Gateway", **BODY) as gateway,
        hilspy.open(mem, FIBEX, ecu="Adas", **BODY) as adas,
        hilspy.open(mem, FIBEX, ecu="Radar", **BODY) as radar,
    ):
        gateway.send("VehicleStatus", VALUES["VehicleStatus"])
        with pytest.raises(hilspy.Error, match="SteeringCmd is sent by Adas, not Gateway"):
            gateway.send("SteeringCmd", VALUES["SteeringCmd"])
        with pytest.raises(hilspy.Error, match="RadarObjects is sent by Radar, not Adas"):
            adas.send("RadarObjects", VALUES["RadarObjects"])
        m = adas.recv(timeout=1)
        assert m is not None and m.name == "VehicleStatus" and m["Gear"] == "D"
        # Radar receives nothing: the frame arrives nameless, with its bytes.
        m = radar.recv(timeout=1)
        assert m is not None and (m.name, m.id, m.fields) == (None, 0x100, {})
        radar.send("RadarObjects", VALUES["RadarObjects"])
        m = adas.recv(timeout=1)
        assert m is not None and m.name == "RadarObjects"
        m = gateway.recv(timeout=1)
        assert m is not None and (m.name, m.id) == (None, 0x200)
        adas.send("SteeringCmd", VALUES["SteeringCmd"])
        m = gateway.recv(timeout=1)
        assert m is not None and m.name == "SteeringCmd"
    # On the Chassis channel the Gateway receives WheelSpeeds from Abs.
    with (
        hilspy.open(mem, FIBEX, channel="Chassis", ecu="Abs") as abs_ecu,
        hilspy.open(mem, FIBEX, channel="Chassis", ecu="Gateway") as gateway,
    ):
        wheels = {"Front.Left": 1.0, "Front.Right": 2.0, "Rear.Left": 3.0, "Rear.Right": 4.0}
        abs_ecu.send("WheelSpeeds", wheels)
        m = gateway.recv(timeout=1)
        assert m is not None and m.name == "WheelSpeeds" and m.fields == wheels
    with pytest.raises(hilspy.Error, match=r'no ECU "Engine" \(ECUs: Gateway, Adas, Radar, Abs\)'):
        hilspy.open(mem, FIBEX, ecu="Engine", **BODY)


def test_the_dbc_names_the_same_ecus(mem: str) -> None:
    with (
        hilspy.open(mem, DBC, ecu="Gateway") as gateway,
        hilspy.open(mem, FIBEX, ecu="Adas", **BODY) as adas,
    ):
        with pytest.raises(hilspy.Error, match="SteeringCmd is sent by Adas, not Gateway"):
            gateway.send("SteeringCmd", VALUES["SteeringCmd"])
        gateway.send("VehicleStatus", VALUES["VehicleStatus"])
        m = adas.recv(timeout=1)
        assert m is not None and m.name == "VehicleStatus"
        adas.send("SteeringCmd", VALUES["SteeringCmd"])
        m = gateway.recv(timeout=1)
        assert m is not None and m.name == "SteeringCmd"


FEATURES = f"fibex:{DATA / 'fibex_features.xml'}"


def round_trip(mem: str, name: str, values: dict[str, Any]) -> tuple[bytes, Message]:
    """Send `values` through the features fixture: the payload and the decoded message."""
    with (
        hilspy.open(mem, FEATURES) as tx,
        hilspy.open(mem) as raw,
        hilspy.open(mem, FEATURES) as rx,
    ):
        tx.send(name, values)
        frame, decoded = raw.recv(timeout=1), rx.recv(timeout=1)
        assert frame is not None and frame.data is not None and decoded is not None
        return frame.data, decoded


def test_sign_magnitude_and_ones_complement(mem: str) -> None:
    values = {"SmSmall": -5, "OcSmall": -5, "SmWide": -100.5, "OcWide": -2}
    data, decoded = round_trip(mem, "Encodings", values)
    assert data == bytes([0x85, 0xFA, 0xC9, 0x80, 0, 0, 0xFF, 0xFD])
    assert decoded.fields == values
    edges = {"SmSmall": -127, "OcSmall": 127, "SmWide": 16383.5, "OcWide": -32767}
    data, decoded = round_trip(f"{mem}-edges", "Encodings", edges)
    assert data == bytes([0xFF, 0x7F, 0xFF, 0x7F, 0, 0, 0x80, 0x00])
    assert decoded.fields == edges
    with hilspy.open(mem, FEATURES) as tx, pytest.raises(hilspy.Error, match="sign-magnitude"):
        tx.send("Encodings", {**values, "SmSmall": -128})


def test_texttable_ranges_piecewise_and_rational_codings(mem: str) -> None:
    values = {
        "Level": "High",
        "Parabola": "NotAvailable",
        "Curve": 90.0,
        "Inverse": 100.0,
        "Ratio": 20.0,
    }
    data, decoded = round_trip(mem, "Codings", values)
    # Level High -> 4 (its lower limit), NotAvailable -> 15; Curve 2x - 150 = 90
    # at 120; Inverse 1000 / (x + 1) = 100 at 9; Ratio (10 + 3x) / 2 = 20 at 10.
    assert data == bytes([0xF4, 120, 9, 10, 0, 0, 0, 0])
    assert decoded.fields == values
    with (
        hilspy.open(mem, FEATURES) as tx,
        hilspy.open(mem) as raw,
        hilspy.open(mem, FEATURES) as rx,
    ):
        # Any raw value of a texttable range decodes to its name (Level 14).
        raw.send(Message(data=bytes([0x3E, 99, 0, 0, 0, 0, 0, 0]), id=0x401))
        m = rx.recv(timeout=1)
        assert m is not None
        assert m.fields == {
            "Level": "High",
            "Parabola": 25.0,
            "Curve": 49.5,
            "Inverse": 1000.0,
            "Ratio": 5.0,
        }
        with pytest.raises(hilspy.Error, match="signal Parabola.*monotonic"):
            tx.send("Codings", {**values, "Parabola": 36.0})
        with pytest.raises(hilspy.Error, match="signal Inverse.*outside"):
            tx.send("Codings", {**values, "Inverse": 2000.0})


FEATURES_DBC = DATA / "fibex_features.dbc"


@pytest.mark.parametrize("name", ["Shifted", "Swapped"])
def test_features_match_the_dbc(mem: str, name: str) -> None:
    """Frames of the features fixture give the payloads of the equivalent DBC (and cantools)."""
    rng = random.Random(name)
    reference: Any = cantools.database.load_file(str(FEATURES_DBC))
    theirs = reference.get_message_by_name(name)
    raw: dict[str, Any] = {"scaling": False, "strict": False, "decode_choices": False}
    fibex_bus, dbc_bus = f"{mem}-fibex", f"{mem}-dbc"
    with (
        hilspy.open(fibex_bus, FEATURES, **raw) as fibex,
        hilspy.open(fibex_bus) as fibex_raw,
        hilspy.open(dbc_bus, f"dbc:{FEATURES_DBC}", **raw) as dbc,
        hilspy.open(dbc_bus) as dbc_raw,
    ):
        for _ in range(50):
            values = random_raw(theirs, rng)
            fibex.send(name, values)
            dbc.send(name, values)
            a, b = fibex_raw.recv(timeout=1), dbc_raw.recv(timeout=1)
            assert a is not None and b is not None
            assert a.data == b.data == theirs.encode(values, scaling=False, strict=False)
            fibex_raw.send(b)
            back = fibex.recv(timeout=1)
            assert back is not None and back.fields == values


MULTIPLEXED: dict[int, dict[str, Any]] = {
    1: {"Mux": 1, "A16": 0x1234, "A8": 0xAB, "Counter": 7},
    2: {"Mux": 2, "B16": 0xBEEF, "B4": 9, "Counter": 8},
    3: {"Mux": 3, "B16": 1, "B4": 15, "Counter": 9},
}


@pytest.mark.parametrize("code", list(MULTIPLEXED))
def test_multiplexed_pdu_matches_the_dbc(mem: str, code: int) -> None:
    """Every switch code gives the payload of the DBC (`M`, `mN` and `SG_MUL_VAL_`)."""
    values = MULTIPLEXED[code]
    dbc_spec = f"dbc:{FEATURES_DBC}"
    fibex_bus, dbc_bus = f"{mem}-fibex", f"{mem}-dbc"
    with (
        hilspy.open(fibex_bus, FEATURES) as fibex,
        hilspy.open(fibex_bus) as fibex_raw,
        hilspy.open(dbc_bus, dbc_spec) as dbc,
        hilspy.open(dbc_bus) as dbc_raw,
    ):
        fibex.send("Multiplexed", values)
        dbc.send("Multiplexed", values)
        a, b = fibex_raw.recv(timeout=1), dbc_raw.recv(timeout=1)
        assert a is not None and b is not None and a.data == b.data
        assert a.data is not None and a.data[0] == code
        # Each decodes the other's frame to the same fields.
        fibex_raw.send(b)
        dbc_raw.send(a)
        from_dbc, from_fibex = fibex.recv(timeout=1), dbc.recv(timeout=1)
        assert from_dbc is not None and from_fibex is not None
        assert from_dbc.fields == from_fibex.fields == values
