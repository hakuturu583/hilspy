"""The sbf codec over mem, file, UDP and TCP."""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

import hilspy
from hilspy import Message

PVT = {
    "tow": 345_600_000,
    "wnc": 2380,
    "mode": 4,
    "latitude": math.radians(35.6895),
    "longitude": math.radians(139.6917),
    "height": 40.0,
    "vn": 1.5,
    "nr_sv": 18,
}


def test_encode_decode(mem: str) -> None:
    with hilspy.open(mem, "sbf") as tx, hilspy.open(mem, "sbf") as rx:
        tx.send("PVTGeodetic", PVT)
        tx.send("AttEuler", {"heading": 90.25, "mode": 4})
        pvt, att = rx.recv(timeout=1), rx.recv(timeout=1)
        assert pvt is not None and pvt.name == "PVTGeodetic"
        assert pvt["tow"] == 345_600_000 and pvt["mode_name"] == "RTK Fixed"
        assert pvt["latitude"] == math.radians(35.6895) and pvt["undulation"] is None
        assert pvt["revision"] == 2 and len(pvt.data or b"") == 96
        assert att is not None and att["heading"] == 90.25 and att["heading_dot"] is None
        # tow/wnc default to the current GPS time.
        assert att["wnc"] > 2300
        with pytest.raises(hilspy.Error, match="no field"):
            tx.send("PVTGeodetic", {"typo": 1})
        with pytest.raises(hilspy.Error, match="cannot encode"):
            tx.send("MeasEpoch", {})


def test_stream_reassembly_with_noise(mem: str) -> None:
    with hilspy.open(mem, "sbf") as enc, hilspy.open(mem) as raw:
        enc.send("PVTGeodetic", PVT)
        block = raw.recv(timeout=1)
        assert block is not None and block.data is not None
        good = block.data
    corrupt = bytearray(good)
    corrupt[20] ^= 0xFF
    stream = b"$GPGGA,noise*00\r\n$" + good + b"\x00$" + bytes(corrupt) + good
    with hilspy.open(mem) as tx, hilspy.open(mem, "sbf") as rx:
        for i in range(0, len(stream), 5):
            tx.send(Message(data=stream[i : i + 5]))
        got = [rx.recv(timeout=1), rx.recv(timeout=1)]
        assert all(m is not None and m.data == good for m in got)
        assert rx.recv(timeout=0.05) is None


def test_replay_a_recording(tmp_path: Path, mem: str) -> None:
    path = tmp_path / "log.sbf"
    with hilspy.open(f"file://{path}?write", "sbf") as rec:
        for t in range(0, 1000, 100):
            rec.send("PVTGeodetic", {**PVT, "tow": t})
    with hilspy.open(f"file://{path}", "sbf") as replay:
        assert [m["tow"] for m in replay] == list(range(0, 1000, 100))


def test_tcp_server_to_client(free_port: int) -> None:
    with (
        hilspy.open(f"tcp://127.0.0.1:{free_port}?listen", "sbf") as receiver_sim,
        hilspy.open(f"tcp://127.0.0.1:{free_port}", "sbf") as client,
    ):
        # The server sends to the clients connected so far.
        deadline = time.monotonic() + 5
        while (m := client.recv(timeout=0.05)) is None:
            receiver_sim.send("AttEuler", {"heading": 42.0})
            assert time.monotonic() < deadline, "no block received"
        assert m.name == "AttEuler" and m["heading"] == 42.0


def test_udp(free_port: int) -> None:
    with (
        hilspy.open(f"udp://127.0.0.1:{free_port}?listen", "sbf") as server,
        hilspy.open(f"udp://127.0.0.1:{free_port}", "sbf") as client,
    ):
        client.send("PVTGeodetic", PVT)
        m = server.recv(timeout=2)
        assert m is not None and m["nr_sv"] == 18
        server.send("AttEuler", {"roll": 1.5})  # replies to the last sender
        r = client.recv(timeout=2)
        assert r is not None and r["roll"] == 1.5


def test_received_messages_can_be_sent_on(mem: str) -> None:
    with hilspy.open(mem, "sbf") as tx, hilspy.open(mem, "sbf") as rx:
        tx.send("PVTGeodetic", {**PVT, "mode_2d": True})
        tx.send("AttEuler", {"heading": 1.0, "mode": 3})
        pvt, att = rx.recv(timeout=1), rx.recv(timeout=1)
        assert pvt is not None and pvt["mode_2d"] is True and pvt["mode"] == 4
        assert att is not None and (att["mode"], att["mode_name"]) == (3, "HeadingPitchRollFloat")
        # Forward them unchanged: the same bytes go out again.
        rx.send(pvt)
        rx.send(att)
        again = [tx.recv(timeout=1), tx.recv(timeout=1)]
        assert [m.data for m in again if m is not None] == [pvt.data, att.data]


def test_close_drops_pending_messages(mem: str) -> None:
    with hilspy.open(mem, "sbf") as enc, hilspy.open(mem) as raw:
        enc.send("AttEuler", {"heading": 1.0})
        enc.send("AttEuler", {"heading": 2.0})
        both = b"".join(m.data or b"" for m in (raw.recv(timeout=1), raw.recv(timeout=1)) if m)
    with hilspy.open(mem) as tx, hilspy.open(mem, "sbf") as rx:
        tx.send(Message(data=both))  # one packet, two blocks
        assert rx.recv(timeout=1) is not None
        rx.close()
        with pytest.raises(hilspy.Error, match="closed"):
            rx.recv(timeout=0)
        assert list(rx) == []


def test_tcp_server_keeps_client_streams_apart(free_port: int, mem: str) -> None:
    with hilspy.open(mem, "sbf") as enc, hilspy.open(mem) as raw:
        enc.send("AttEuler", {"heading": 1.0})
        enc.send("AttEuler", {"heading": 2.0})
        a, b = (m.data or b"" for m in (raw.recv(timeout=1), raw.recv(timeout=1)) if m)
    with (
        hilspy.open(f"tcp://127.0.0.1:{free_port}?listen", "sbf") as server,
        hilspy.open(f"tcp://127.0.0.1:{free_port}") as client_a,
        hilspy.open(f"tcp://127.0.0.1:{free_port}") as client_b,
    ):
        # Interleave on the server: half of A, all of B, rest of A.
        client_a.send(Message(data=a[:20]))
        time.sleep(0.1)
        client_b.send(Message(data=b))
        time.sleep(0.1)
        client_a.send(Message(data=a[20:]))
        got = sorted(m["heading"] for m in (server.recv(timeout=2), server.recv(timeout=2)) if m)
        assert got == [1.0, 2.0]
