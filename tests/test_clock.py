"""Channels on a virtual clock: sends happen as the simulation steps it."""

from __future__ import annotations

import math
import time

import pytest

import hilspy


def drain(ch: hilspy.Channel) -> list[hilspy.Message]:
    out = []
    while (m := ch.recv(timeout=0.05)) is not None:
        out.append(m)
    return out


def test_periodic_messages_follow_the_clock(mem: str, vehicle_dbc: str) -> None:
    clock = hilspy.Clock(start=10.0)
    assert clock.now == 10.0 and clock.next_due is None
    with (
        hilspy.open(mem, vehicle_dbc, clock=clock, strict=False) as ecu,
        hilspy.open(mem, vehicle_dbc, clock=clock) as rx,
    ):
        speed = ecu.every(0.01, "VehicleSpeed", {"Speed": 1.0, "Gear": "D"})
        steer = ecu.every(0.02, "SteeringCmd", {"SteeringAngle": 2.0})
        # The first sends happen at once; nothing else in wall time.
        time.sleep(0.05)
        assert (speed.sent_count, steer.sent_count) == (1, 1)
        assert clock.next_due == pytest.approx(10.01)
        # One 50 ms simulation step sends what falls due in it.
        clock.advance(0.05)
        assert (speed.sent_count, steer.sent_count) == (6, 3)
        got = drain(rx)
        assert [m.name for m in got].count("VehicleSpeed") == 6
        assert all(m.timestamp == pytest.approx(10.05) for m in got)
        speed.update("VehicleSpeed", {"Speed": 2.0, "Gear": "D"})
        steer.stop()
        clock.advance_to(10.1)
        got = drain(rx)
        assert [m.name for m in got] == ["VehicleSpeed"] * 5
        assert got[-1]["Speed"] == 2.0
        assert speed.running and not steer.running


def test_duration_and_errors(mem: str) -> None:
    clock = hilspy.Clock()
    with hilspy.open(mem, clock=clock) as ch:
        p = ch.every(0.1, hilspy.Message(data=b"x"), duration=0.25)
        clock.advance(1.0)
        assert p.sent_count == 3 and not p.running
    with pytest.raises(hilspy.Error, match="back"):
        clock.advance_to(0.5)
    with pytest.raises(hilspy.Error):
        clock.advance(-0.1)
    with pytest.raises(hilspy.Error):
        hilspy.Clock(math.nan)


def test_lidar_revolution_spreads_over_steps(mem: str) -> None:
    clock = hilspy.Clock(start=1_760_000_000.0)
    ring = {
        "x": [10 * math.cos(math.radians(a)) for a in range(360)],
        "y": [-10 * math.sin(math.radians(a)) for a in range(360)],
        "z": [0.0] * 360,
    }
    with (
        hilspy.open(mem, "lidar:pandar_xt32", clock=clock) as sensor,
        hilspy.open(mem, "lidar:pandar_xt32", frame="packet") as driver,
    ):
        revolutions = sensor.every(0.1, "scan", ring)
        assert len(drain(driver)) == 1
        # Simulation steps of 20 ms: a fifth of the 250 packets each.
        counts = []
        for _ in range(5):
            clock.advance(0.02)
            counts.append(len(drain(driver)))
        # Packets every 0.4 ms; the next revolution starts at 0.1 s, in step 5.
        assert counts == [50, 50, 50, 50, 50]
        assert revolutions.sent_count == 2


def test_sbf_time_is_the_clock(mem: str) -> None:
    # 2024-01-01T00:00:00Z: GPS week 2295, TOW Monday 00:00:18.
    clock = hilspy.Clock(start=1_704_067_200.0)
    with hilspy.open(mem, "sbf", clock=clock) as rx, hilspy.open(mem, "sbf", clock=clock) as gnss:
        attitude = gnss.every(0.1, "AttEuler", {"heading": 90.0})
        clock.advance(0.2)
        tows = [m["tow"] for m in drain(rx)]
        assert tows == [86_418_000, 86_418_100, 86_418_200]
        assert attitude.sent_count == 3
