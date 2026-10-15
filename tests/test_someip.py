"""The someip codec: SOME/IP methods, events and fields of a FIBEX 4 service."""

from __future__ import annotations

import array
import struct
import time
from pathlib import Path
from typing import Any

import pytest

import hilspy
from hilspy import Message

DATA = Path(__file__).parent / "data"
SOMEIP = f"someip:{DATA / 'someip.xml'}"

ORIGIN = {"latitude": 35.6895, "longitude": 139.6917, "altitude": 40.25, "quality": "RTK_FIXED"}
ROUTE: dict[str, Any] = {
    "origin": ORIGIN,
    "waypoints": [{"id": 1, "speed": -20}, {"id": 2, "speed": 300}],
    "name": "Route 1",
}
STATUS: dict[str, Any] = {
    "speed": -1234,
    "timestamp": 1_700_000_000_123,
    "position": {**ORIGIN, "quality": "GNSS"},
}
HEADER = (
    "service_id",
    "method_id",
    "length",
    "client_id",
    "session_id",
    "protocol_version",
    "interface_version",
    "message_type",
    "return_code",
)


def params(m: Message) -> dict[str, Any]:
    return {k: v for k, v in m.fields.items() if k not in HEADER}


def test_request_response_over_mem(mem: str) -> None:
    with (
        hilspy.open(mem, SOMEIP, client_id=0x42) as client,
        hilspy.open(mem, SOMEIP) as server,
    ):
        client.send("VehicleService.SetRoute", ROUTE)
        req = server.recv(timeout=1)
        assert req is not None and req.name == "VehicleService.SetRoute"
        assert params(req) == ROUTE
        assert (req["service_id"], req["method_id"], req["client_id"]) == (0x1234, 1, 0x42)
        assert (req["session_id"], req["message_type"]) == (1, "REQUEST")
        assert req["return_code"] == "E_OK"
        assert (req["protocol_version"], req["interface_version"]) == (1, 1)
        assert req.data is not None and req["length"] == len(req.data) - 8
        length = struct.pack(">I", req["length"])
        rest = bytes.fromhex("0042 0001 01010000")
        assert req.data[:16] == bytes.fromhex("1234 0001") + length + rest

        # The response echoes the request's client and session IDs.
        server.send(
            "VehicleService.SetRoute",
            {
                "message_type": "RESPONSE",
                "client_id": req["client_id"],
                "session_id": req["session_id"],
                "accepted": 1,
                "eta": [10, 20, 30],
                "distances": [1.5, -2.25],
            },
        )
        resp = client.recv(timeout=1)
        assert resp is not None and resp["message_type"] == "RESPONSE"
        assert (resp["client_id"], resp["session_id"]) == (0x42, 1)
        assert resp["eta"] == array.array("I", [10, 20, 30])
        assert resp["distances"] == array.array("f", [1.5, -2.25])
        assert resp["accepted"] == 1

        # Sessions count per method.
        client.send("VehicleService.SetRoute", ROUTE)
        again = server.recv(timeout=1)
        assert again is not None and again["session_id"] == 2


def test_decoded_messages_encode_to_the_same_bytes(mem: str) -> None:
    with hilspy.open(mem, SOMEIP) as tx, hilspy.open(mem, SOMEIP) as rx:
        tx.send("VehicleService.Status", STATUS)
        tx.send("VehicleService.Configure", {"setting": {"gain": 7}, "label": "abc"})
        tx.send("VehicleService.changed_Mode_field", {"Mode": "GNSS"})
        got = [rx.recv(timeout=1) for _ in range(3)]
        assert [m.name for m in got if m] == [
            "VehicleService.Status",
            "VehicleService.Configure",
            "VehicleService.changed_Mode_field",
        ]
        status, configure, mode = (m for m in got if m)
        assert params(status) == STATUS and status["message_type"] == "NOTIFICATION"
        assert params(configure) == {"setting": {"gain": 7}, "label": "abc"}
        assert mode["Mode"] == "GNSS"
        for m in (status, configure, mode):
            rx.send(m)
            echo = tx.recv(timeout=1)
            assert echo is not None and echo.data == m.data


def test_hand_built_bytes(mem: str) -> None:
    # The Mode field's setter: one enum byte.
    frame = bytes.fromhex("1234 0011 00000009 abcd 0005 01 01 00 00 02")
    with hilspy.open(mem) as raw, hilspy.open(mem, SOMEIP, decode_choices=False) as rx:
        raw.send(Message(data=frame))
        m = rx.recv(timeout=1)
        assert m is not None and m.name == "VehicleService.set_Mode_field"
        assert m.fields == {
            "service_id": 0x1234,
            "method_id": 0x11,
            "length": 9,
            "client_id": 0xABCD,
            "session_id": 5,
            "protocol_version": 1,
            "interface_version": 1,
            "message_type": 0,
            "return_code": 0,
            "Mode": 2,
        }


def test_several_messages_in_one_packet_and_unknown_services(mem: str) -> None:
    with hilspy.open(mem, SOMEIP) as tx, hilspy.open(mem) as raw:
        tx.send("VehicleService.Status", STATUS)
        tx.send("VehicleService.get_Mode_field")
        first, second = raw.recv(timeout=1), raw.recv(timeout=1)
        assert first is not None and second is not None
        unknown = bytes.fromhex("ffff 8100 0000000c 0000 0001 01 01 02 00 00000000")
    with hilspy.open(mem) as tx, hilspy.open(mem, SOMEIP) as rx:
        tx.send(Message(data=(first.data or b"") + (second.data or b"") + unknown))
        got = [rx.recv(timeout=1) for _ in range(3)]
        assert [m.name for m in got if m] == [
            "VehicleService.Status",
            "VehicleService.get_Mode_field",
            None,
        ]
        assert got[2] is not None and got[2].data == unknown


def test_errors(mem: str) -> None:
    with hilspy.open(mem, SOMEIP) as ch:
        with pytest.raises(hilspy.Error, match="no SOME/IP method"):
            ch.send("VehicleService.Nope", {})
        with pytest.raises(hilspy.Error, match="missing parameter"):
            ch.send("VehicleService.SetRoute", {"origin": ORIGIN})
        with pytest.raises(hilspy.Error, match="out of range"):
            ch.send("VehicleService.Status", {**STATUS, "speed": 40_000})
    with pytest.raises(hilspy.Error, match="does not take"):
        hilspy.open(mem, SOMEIP, typo=1)
    with pytest.raises(hilspy.Error, match="16-bit"):
        hilspy.open(mem, SOMEIP, client_id=-1)
    # A truncated payload fails to decode.
    with hilspy.open(mem) as tx, hilspy.open(mem, SOMEIP) as rx:
        tx.send(Message(data=bytes.fromhex("1234 8001 00000009 0000 0001 01 01 02 00 00")))
        with pytest.raises(hilspy.Error, match="truncated"):
            rx.recv(timeout=1)


def test_udp_loopback(free_port: int) -> None:
    with (
        hilspy.open(f"udp://127.0.0.1:{free_port}?listen", SOMEIP) as server,
        hilspy.open(f"udp://127.0.0.1:{free_port}", SOMEIP, client_id=7) as client,
    ):
        client.send("VehicleService.SetRoute", ROUTE)
        req = server.recv(timeout=2)
        assert req is not None and params(req) == ROUTE and req["client_id"] == 7
        server.send("VehicleService.Status", STATUS)  # to the last sender
        event = client.recv(timeout=2)
        assert event is not None and event.name == "VehicleService.Status"
        assert params(event) == STATUS


def test_tcp_stream(free_port: int) -> None:
    with (
        hilspy.open(f"tcp://127.0.0.1:{free_port}?listen", SOMEIP) as server,
        hilspy.open(f"tcp://127.0.0.1:{free_port}", SOMEIP) as client,
    ):
        deadline = time.monotonic() + 5
        while (m := client.recv(timeout=0.05)) is None:
            server.send("VehicleService.changed_Mode_field", {"Mode": "NONE"})
            assert time.monotonic() < deadline, "no event received"
        assert m.name == "VehicleService.changed_Mode_field" and m["Mode"] == "NONE"
