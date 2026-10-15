"""LIN and FlexRay frames: Message headers, mem:// and the fibex codec."""

from __future__ import annotations

from pathlib import Path

import pytest

import hilspy
from hilspy import Message


def test_lin_message() -> None:
    m = Message(data=bytes([0x4A, 0x55, 0x93, 0xE5]), id=0x10, bus="lin")
    assert (m.bus, m.id, m.enhanced, m.extended, m.fd) == ("lin", 0x10, True, False, False)
    # Enhanced: over the protected id 0x50 and the data (hand-computed).
    assert m.checksum == 0xFF - 0x69
    classic = Message(data=bytes([0x4A, 0x55, 0x93, 0xE5]), id=0x10, enhanced=False)
    assert (classic.bus, classic.enhanced, classic.checksum) == ("lin", False, 0xFF - 0x19)
    # Diagnostic frames default to (and always compute) classic.
    diag = Message(data=bytes(8), id=0x3C, bus="lin")
    assert not diag.enhanced and diag.checksum == 0xFF
    assert Message(data=b"\x01", id=1, checksum=0x12).checksum == 0x12
    assert m.cycle is None and m.channel is None
    assert 'bus="lin"' in repr(m)


def test_flexray_message() -> None:
    m = Message(data=bytes(4), id=5, bus="flexray", cycle=3, channel="B")
    assert (m.bus, m.id, m.cycle, m.channel) == ("flexray", 5, 3, "B")
    assert m.checksum is None and not m.enhanced
    d = Message(data=bytes(2), id=7, cycle=0)
    assert (d.bus, d.channel) == ("flexray", "A")
    assert Message(data=bytes(2), id=7, channel="ab").channel == "AB"
    can = Message(data=b"", id=0x123)
    assert (can.bus, can.cycle, can.checksum) == ("can", None, None)
    assert Message(data=b"x").bus is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": 64, "bus": "lin"},
        {"id": 1, "bus": "lin", "data": bytes(9)},
        {"id": 1, "bus": "lin", "fd": True},
        {"id": 1, "enhanced": True, "cycle": 1},
        {"id": 0, "bus": "flexray"},
        {"id": 2048, "bus": "flexray"},
        {"id": 1, "cycle": 64},
        {"id": 1, "cycle": 0, "data": b"odd"},
        {"id": 1, "channel": "C"},
        {"id": 1, "bus": "most"},
        {"bus": "lin", "data": b""},
        {"id": 0x123, "cycle": 1, "bus": "can"},
    ],
)
def test_header_validation(kwargs: dict[str, object]) -> None:
    kwargs.setdefault("data", b"")
    with pytest.raises(ValueError):
        Message(**kwargs)  # type: ignore[arg-type]


def test_headers_travel_over_mem(mem: str) -> None:
    with hilspy.open(mem) as a, hilspy.open(mem) as b:
        a.send(Message(data=b"\x01\x02", id=0x21, bus="lin", checksum=0x00))
        a.send(Message(data=b"\x01\x02", id=12, cycle=33, channel="AB"))
        lin = b.recv(timeout=1)
        assert lin is not None
        assert (lin.bus, lin.id, lin.enhanced, lin.checksum) == ("lin", 0x21, True, 0)
        fr = b.recv(timeout=1)
        assert fr is not None
        assert (fr.bus, fr.id, fr.cycle, fr.channel) == ("flexray", 12, 33, "AB")
        assert fr.data == b"\x01\x02"


DATA = Path(__file__).parent / "data"
FIBEX = f"fibex:{DATA / 'lin_flexray.xml'}"
LIGHT_CMD = {"Brightness": 50.0, "Mode": "High"}


def drain(ch: hilspy.Channel) -> list[Message]:
    out = []
    while (m := ch.recv(timeout=0.05)) is not None:
        out.append(m)
    return out


def test_lin_round_trip_over_mem(mem: str) -> None:
    with (
        hilspy.open(mem, FIBEX, channel="Lin", ecu="Bcm") as master,
        hilspy.open(mem, FIBEX, channel="Lin", ecu="Lamp") as slave,
        hilspy.open(mem) as raw,
    ):
        master.send("LightCmd", LIGHT_CMD)
        frame = raw.recv(timeout=1)
        assert frame is not None
        assert (frame.bus, frame.id, frame.enhanced, frame.checksum) == ("lin", 0x10, True, 0x49)
        assert frame.data == b"\x64\x02"
        got = slave.recv(timeout=1)
        assert got is not None and got.name == "LightCmd" and got.fields == LIGHT_CMD
        assert got.bus == "lin" and got.checksum == 0x49
        slave.send("LightStatus", {"Current": 1.5, "Fault": 0})
        status = master.recv(timeout=1)
        assert status is not None and status.name == "LightStatus"
        assert (status.enhanced, status.checksum) == (False, 0xF0)


def test_lin_bad_checksum_is_a_decode_error(mem: str) -> None:
    with hilspy.open(mem, FIBEX, channel="Lin") as rx, hilspy.open(mem) as tx:
        tx.send(Message(data=b"\x64\x02", id=0x10, checksum=0x48))  # 0x49 is right
        with pytest.raises(
            hilspy.Error, match=r"checksum 0x48, expected the enhanced checksum 0x49"
        ):
            rx.recv(timeout=1)
        # A classic checksum on an enhanced frame is wrong too.
        tx.send(Message(data=b"\x64\x02", id=0x10, enhanced=False))
        with pytest.raises(hilspy.Error, match="LIN frame LightCmd"):
            rx.recv(timeout=1)
        # The stream goes on.
        tx.send(Message(data=b"\x64\x02", id=0x10, bus="lin"))
        ok = rx.recv(timeout=1)
        assert ok is not None and ok.name == "LightCmd"
        tx.send(Message(data=b"\x01", id=0x20, bus="lin"))
        unknown = rx.recv(timeout=1)
        assert unknown is not None and unknown.name is None and unknown.id == 0x20


def test_lin_schedule_cycle_times_on_a_clock(mem: str) -> None:
    clock = hilspy.Clock()
    with (
        hilspy.open(mem, FIBEX, channel="Lin", clock=clock) as master,
        hilspy.open(mem, FIBEX, channel="Lin") as rx,
    ):
        # Schedule table Normal: 40 ms, LightCmd twice, LightStatus once.
        cmd = master.every(None, "LightCmd", LIGHT_CMD)
        status = master.every(None, "LightStatus", {"Current": 0.5, "Fault": 1})
        assert (cmd.period, status.period) == pytest.approx((0.02, 0.04))
        clock.advance(0.1)
        assert (cmd.sent_count, status.sent_count) == (6, 3)
        names = [m.name for m in drain(rx)]
        assert (names.count("LightCmd"), names.count("LightStatus")) == (6, 3)
    with (
        hilspy.open(mem, FIBEX, channel="Lin", schedule="Diagnostic", clock=clock) as diag,
    ):
        req = diag.every(None, "MasterReq", {"Nad": 0x7F, "Pci": 6, "Sid": 0xB2})
        assert req.period == pytest.approx(0.02)
        with pytest.raises(hilspy.Error, match='"LightCmd" has no cycle time'):
            diag.every(None, "LightCmd", LIGHT_CMD)


BRAKE = {"BrakeTorque": 100.0, "Counter": 1}
YAW = {"YawRate": 1.5, "Valid": 1}


def test_flexray_round_trip_over_mem(mem: str) -> None:
    with (
        hilspy.open(mem, FIBEX, channel="FlexRayA") as a,
        hilspy.open(mem, FIBEX, channel="FlexRayA", ecu="Esc") as esc,
        hilspy.open(mem, FIBEX, channel="FlexRayB") as b,
        hilspy.open(mem) as raw,
    ):
        a.send("YawFr", YAW)
        frame = raw.recv(timeout=1)
        assert frame is not None
        assert (frame.bus, frame.id, frame.channel) == ("flexray", 7, "A")
        assert frame.cycle is not None and frame.cycle % 4 == 1
        got = esc.recv(timeout=1)
        assert got is not None and got.name == "YawFr" and got.fields == YAW
        assert (got.id, got.cycle) == (7, frame.cycle)
        # Channel B decodes nothing sent on A only.
        unknown = b.recv(timeout=1)
        assert unknown is not None and unknown.name is None and unknown.channel == "A"


def test_flexray_wrong_cycle_does_not_match(mem: str) -> None:
    with hilspy.open(mem, FIBEX, channel="FlexRayA") as rx, hilspy.open(mem) as tx:
        payload = bytes([150, 0, 1, 0])
        for cycle in (5, 7, 6):  # YawFr (base 1), DiagFr (base 3), neither
            tx.send(Message(data=payload, id=7, cycle=cycle, channel="A"))
        names = [m.name for m in drain(rx)]
        assert names == ["YawFr", "DiagFr", None]
        tx.send(Message(data=payload, id=7, cycle=1, channel="AB"))
        both = rx.recv(timeout=1)
        assert both is not None and both.name == "YawFr" and both["YawRate"] == 1.5


def test_flexray_cycle_times_on_a_clock(mem: str) -> None:
    clock = hilspy.Clock()
    with (
        hilspy.open(mem, FIBEX, channel="FlexRayA", clock=clock) as tx,
        hilspy.open(mem, FIBEX, channel="FlexRayA") as rx,
    ):
        # Cycle 5 ms: BrakeFr every cycle, YawFr every 4th from cycle 1.
        brake = tx.every(None, "BrakeFr", BRAKE)
        yaw = tx.every(None, "YawFr", YAW)
        assert (brake.period, yaw.period) == pytest.approx((0.005, 0.02))
        clock.advance(0.04)
        assert (brake.sent_count, yaw.sent_count) == (9, 3)
        got = drain(rx)
        assert [m.cycle for m in got if m.name == "BrakeFr"] == list(range(9))
        assert [m.cycle for m in got if m.name == "YawFr"] == [1, 5, 9]
