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
