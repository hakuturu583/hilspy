"""The lidar codec: Hesai/Velodyne data packets <-> point clouds."""

from __future__ import annotations

import math
from array import array
from pathlib import Path

import numpy as np
import pytest

import hilspy

PROFILES = Path(__file__).parents[1] / "crates/hilspy-core/lidar/profiles.binpb"
MODELS = [
    "pandar_40p",
    "pandar_64",
    "pandar_qt64",
    "pandar_qt128",
    "pandar_xt16",
    "pandar_xt32",
    "pandar_xt32m",
    "pandar_128e3x",
    "pandar_128e4x",
    "pandar_at128",
    "vlp16",
    "vlp16_hires",
    "vlp32c",
    "vlp32mr",
    "hdl32e",
    "hdl64e",
    "vls128",
]


def ring(n: int = 720, distance: float = 10.0) -> dict[str, list[float]]:
    """Points `distance` m away on the horizon, `n` per revolution."""
    angles = [2 * math.pi * i / n for i in range(n)]
    return {
        "x": [distance * math.cos(a) for a in angles],
        "y": [-distance * math.sin(a) for a in angles],
        "z": [0.0] * n,
    }


def drain(ch: hilspy.Channel) -> list[hilspy.Message]:
    out = []
    while (m := ch.recv(timeout=0.5)) is not None:
        out.append(m)
    return out


@pytest.mark.parametrize("model", MODELS)
def test_every_model_round_trips(mem: str, model: str) -> None:
    with (
        hilspy.open(mem, f"lidar:{model}") as tx,
        hilspy.open(mem, f"lidar:{model}", frame="packet") as rx,
    ):
        # 2 degrees apart: coarser than any sensor's resolution (QT64: 0.6).
        tx.send("scan", {**ring(180), "intensity": 9, "timestamp": 1_760_000_000.0})
        packets = drain(rx)
        assert packets and all(m.name == "scan" for m in packets)
        x = [v for m in packets for v in m["x"]]
        # AT128 sees 120 degrees of the ring; the rest a whole revolution.
        assert len(x) == (60 if model == "pandar_at128" else 180), model
        distance = [v for m in packets for v in m["distance"]]
        assert all(abs(d - 10.0) < 0.2 for d in distance)
        assert {v for m in packets for v in m["intensity"]} == {9}
        assert all(m["model"] == model for m in packets)
        assert packets[0].timestamp == packets[0]["timestamp"]
        t = packets[0]["timestamp"]
        if model.startswith(("vlp", "hdl", "vls")):
            # Velodyne packets carry microseconds past the hour only.
            t = 1_760_000_000.0 + (t - 1_760_000_000.0) % 3600
        assert abs(t - 1_760_000_000.0) < 0.05


def test_scans_and_array_fields(mem: str) -> None:
    with hilspy.open(mem, "lidar:pandar_xt32") as tx, hilspy.open(mem, "lidar:xt32") as rx:
        for rev in range(3):
            tx.send("scan", {**ring(), "timestamp": 100.0 + rev * 0.1})
        scans = drain(rx)
        # A scan ends when the next revolution starts.
        assert len(scans) == 2
        s = scans[1]
        assert abs(s["timestamp"] - 100.1) < 1e-4
        assert s["return_mode"] == "strongest" and s["rpm"] == 600 and s["packets"] == 250
        types = {k: v.typecode for k, v in s.fields.items() if isinstance(v, array)}
        assert types == {
            "x": "f",
            "y": "f",
            "z": "f",
            "intensity": "B",
            "channel": "H",
            "azimuth": "f",
            "distance": "f",
            "return": "B",
            "time": "f",
        }
        assert len(s["x"]) == 720
        assert all(0.0 <= t < 0.1 for t in s["time"])
        # The XT32's channel nearest the horizon.
        assert set(s["channel"]) == {15}
        # numpy reads the arrays without copying them element by element.
        xyz = np.frombuffer(s["x"], dtype=np.float32)
        assert np.allclose(np.hypot(xyz, np.frombuffer(s["y"], dtype=np.float32)), 10, atol=0.01)
        # A received scan can be sent back as is.
        tx.send(s)
        tx.send("scan", {**ring(), "timestamp": 101.0})
        (again,) = drain(rx)[-1:]
        assert sorted(again["azimuth"]) == pytest.approx(sorted(s["azimuth"]), abs=0.01)


def test_numpy_and_polar_input(mem: str) -> None:
    with (
        hilspy.open(mem, "lidar:vlp16", rpm=1200) as tx,
        hilspy.open(mem, "lidar:vlp16", frame="packet") as rx,
    ):
        azimuth = np.arange(0, 360, 1.0, dtype=np.float64)
        tx.send(
            "scan",
            {
                "distance": np.full(360, 25.0, dtype=np.float32),
                "azimuth": azimuth,
                "channel": np.full(360, 3, dtype=np.uint16),
                "intensity": np.arange(360) % 256,
            },
        )
        packets = drain(rx)
        assert len(packets) == 38  # 1200 rpm: 0.4 degrees per firing
        assert {c for m in packets for c in m["channel"]} == {3}
        got = sorted(a for m in packets for a in m["azimuth"])
        assert got == pytest.approx(list(azimuth), abs=0.21)


def test_udp_sensor_and_profile_file(free_port: int) -> None:
    # A simulated sensor sends to port `free_port`; a driver listens there.
    with (
        hilspy.open(f"udp://127.0.0.1:{free_port}?listen", "lidar:VLP-16") as driver,
        hilspy.open(
            f"udp://127.0.0.1:{free_port}", "lidar:vlp16", profiles=str(PROFILES)
        ) as sensor,
    ):
        sensor.send("scan", {**ring(), "timestamp": 3600.0 * 1000 + 12.5})
        sensor.send("scan", {**ring(), "timestamp": 3600.0 * 1000 + 12.6})
        (scan,) = drain(driver)
        assert len(scan["x"]) == 720
        assert scan["rpm"] is None  # VLP-16 packets carry no motor speed
        # Velodyne packets carry microseconds past the hour; the hour comes
        # from the reception time.
        assert scan["timestamp"] % 3600 == pytest.approx(12.5, abs=1e-4)


def test_errors(mem: str, tmp_path: Path) -> None:
    with pytest.raises(hilspy.Error, match="needs a model"):
        hilspy.open(mem, "lidar")
    with pytest.raises(hilspy.Error, match="unknown LiDAR"):
        hilspy.open(mem, "lidar:nope")
    bad = tmp_path / "bad.binpb"
    bad.write_bytes(b"\xff\xff")
    with pytest.raises(hilspy.Error, match="ProfileSet"):
        hilspy.open(mem, "lidar:vlp16", profiles=str(bad))
    with pytest.raises(hilspy.Error, match="single-return"):
        hilspy.open(mem, "lidar:vlp16", return_mode="last_strongest")
    with hilspy.open(mem, "lidar:pandar_xt32") as lidar, hilspy.open(mem) as raw:
        with pytest.raises(hilspy.Error, match="x, y and z"):
            lidar.send("scan", {"x": [1.0]})
        with pytest.raises(hilspy.Error, match="finite"):
            lidar.send("scan", {**ring(), "timestamp": float("nan")})
        with pytest.raises(hilspy.Error, match="numbers"):
            lidar.send("scan", {"x": [1.0, "a"], "y": [0.0], "z": [0.0]})
        raw.send(hilspy.Message(data=b"\x00" * 64))
        with pytest.raises(hilspy.Error, match="not a pandar_xt32 data packet"):
            lidar.recv(timeout=1)


def test_list_values_elsewhere(mem: str) -> None:
    # Arrays are field values for every codec; DBC signals reject them.
    m = hilspy.Message("x", {"a": [1, 2], "b": (1.5, 2), "c": array("h", [-1, 2])})
    assert m["a"] == array("q", [1, 2])
    assert m["b"] == array("d", [1.5, 2.0])
    assert m["c"] == array("i", [-1, 2])
    assert m["c"].typecode == "i"


def test_packet_sizes() -> None:
    # Real sensors: Velodyne 1206-byte, Hesai XT32 1080-byte datagrams.
    for model, size in (("vlp16", 1206), ("pandar_xt32", 1080), ("pandar_40p", 1262)):
        with hilspy.open("mem://size", f"lidar:{model}") as tx, hilspy.open("mem://size") as rx:
            tx.send("scan", ring())
            m = rx.recv(timeout=1)
            assert m is not None and len(m.data or b"") == size, model
            drain(rx)
