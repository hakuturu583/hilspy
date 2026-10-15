"""CAN logs (candump, BLF, ASC) through file://, cross-checked with python-can."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import can
import pytest

import hilspy

FORMATS = [
    pytest.param("drive.blf", "", can.BLFWriter, can.BLFReader, id="blf"),
    pytest.param("drive.asc", "", can.ASCWriter, can.ASCReader, id="asc"),
    pytest.param(
        "drive.log",
        "&format=candump",
        can.CanutilsLogWriter,
        can.CanutilsLogReader,
        id="candump",
        marks=pytest.mark.skipif(sys.platform != "linux", reason="candump logs need Linux"),
    ),
]

FRAMES = [
    (0x123, False, False, b"\x01\x02\x03"),
    (0x18DAF110, True, False, b"\xff" * 8),
    (0x400, False, True, bytes(range(64))),
]


def drain(ch: hilspy.Channel) -> list[hilspy.Message]:
    out = []
    try:
        while (m := ch.recv(timeout=1)) is not None:
            out.append(m)
    except hilspy.Error:
        pass  # the end of the log closes the channel
    return out


@pytest.mark.parametrize(("name", "query", "writer", "reader"), FORMATS)
def test_written_logs_read_in_python_can(tmp_path: Path, name: str, query: str, writer: Any, reader: Any) -> None:
    path = tmp_path / name
    with hilspy.open(f"file://{path}?write{query}") as log:
        for can_id, ext, fd, data in FRAMES:
            log.send(hilspy.Message(data=data, id=can_id, extended=ext, fd=fd))
    got = list(reader(str(path)))
    assert [(m.arbitration_id, m.is_extended_id, m.is_fd, bytes(m.data)) for m in got] == FRAMES


@pytest.mark.parametrize(("name", "query", "writer", "reader"), FORMATS)
def test_python_can_logs_replay(tmp_path: Path, name: str, query: str, writer: Any, reader: Any) -> None:
    path = tmp_path / name
    w = writer(str(path))
    for i, (can_id, ext, fd, data) in enumerate(FRAMES):
        w.on_message_received(
            can.Message(timestamp=1_700_000_000 + i * 0.01, arbitration_id=can_id, is_extended_id=ext, is_fd=fd, data=data)
        )
    w.stop()
    with hilspy.open(f"file://{path}?realtime{query}") as log:
        got = drain(log)
    assert [(m.id, m.extended, m.fd, m.data) for m in got] == FRAMES
    stamps = [m.timestamp or 0.0 for m in got]
    assert stamps[2] - stamps[0] == pytest.approx(0.02, abs=1e-5)


def test_a_log_through_a_dbc(tmp_path: Path, vehicle_dbc: str) -> None:
    path = tmp_path / "speed.blf"
    with hilspy.open(f"file://{path}?write", vehicle_dbc, strict=False) as log:
        log.send("VehicleSpeed", {"Speed": 42.0, "Gear": "D"})
    with hilspy.open(f"file://{path}", vehicle_dbc) as log:
        (m,) = drain(log)
    assert m.name == "VehicleSpeed" and m["Speed"] == 42.0
