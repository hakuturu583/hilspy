"""asyncio and several buses at once."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

import hilspy
from hilspy import Message

DBC = f"dbc:{Path(__file__).parent / 'data' / 'vehicle.dbc'}"


def run(coro: Any) -> Any:
    return asyncio.run(asyncio.wait_for(coro, timeout=30))


def test_asend_arecv(mem: str) -> None:
    async def main() -> None:
        async with hilspy.open(mem) as a, hilspy.open(mem) as b:
            start = time.monotonic()
            assert await a.arecv(timeout=0.1) is None
            assert 0.09 <= time.monotonic() - start < 2
            await b.asend(Message(data=b"hi"))
            m = await a.arecv(timeout=5)
            assert m is not None and m.data == b"hi"

    run(main())


def test_async_iteration_ends_on_close(mem: str) -> None:
    async def main() -> list[bytes | None]:
        rx, tx = hilspy.open(mem), hilspy.open(mem)
        got: list[bytes | None] = []

        async def consume() -> None:
            async for m in rx:
                got.append(m.data)

        task = asyncio.create_task(consume())
        for i in range(3):
            await tx.asend(Message(data=bytes([i])))
        while len(got) < 3:
            await asyncio.sleep(0.01)
        rx.close()
        await task
        with pytest.raises(hilspy.Error, match="closed"):
            await rx.arecv()
        return got

    assert run(main()) == [b"\x00", b"\x01", b"\x02"]


def test_awaiting_does_not_block_the_event_loop(mem: str) -> None:
    async def main() -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        t = asyncio.create_task(ticker())
        async with hilspy.open(mem) as ch:
            assert await ch.arecv(timeout=0.3) is None
        t.cancel()
        return ticks

    assert run(main()) >= 10


def test_merge_several_buses_and_formats(mem: str, free_port: int) -> None:
    """A HIL loop: a DBC CAN bus, SBF over TCP and raw UDP, all at once."""

    async def main() -> dict[str, Any]:
        tcp = f"tcp://127.0.0.1:{free_port}"
        async with (
            hilspy.open(mem, DBC) as can_bus,
            hilspy.open(mem, DBC) as ecu,
            hilspy.open(f"{tcp}?listen", "sbf") as gnss_sim,
            hilspy.open(tcp, "sbf") as gnss,
            hilspy.open("udp://127.0.0.1:0?listen") as udp_rx,
        ):
            seen: dict[str, Any] = {}
            async with hilspy.merge(can_bus, gnss) as buses:

                async def produce() -> None:
                    await ecu.asend(
                        "VehicleSpeed", {"Speed": 10.0, "Accel": 0.0, "Gear": "D", "Counter": 1}
                    )
                    # The TCP server sends to clients connected so far.
                    while "AttEuler" not in seen:
                        await gnss_sim.asend("AttEuler", {"heading": 33.0})
                        await asyncio.sleep(0.02)

                producer = asyncio.create_task(produce())
                async for bus, msg in buses:
                    assert msg.name is not None
                    seen[msg.name] = (bus, msg)
                    if {"VehicleSpeed", "AttEuler"} <= seen.keys():
                        break
                await producer
            assert seen["VehicleSpeed"][0] is can_bus and seen["VehicleSpeed"][1]["Gear"] == "D"
            assert seen["AttEuler"][0] is gnss and seen["AttEuler"][1]["heading"] == 33.0
            # After merge() the channels are usable again.
            assert udp_rx.recv(timeout=0) is None
            await ecu.asend("VehicleSpeed", {"Speed": 5.0, "Accel": 0.0, "Gear": "P", "Counter": 2})
            m = await can_bus.arecv(timeout=5)
            assert m is not None and m["Gear"] == "P"
            return seen

    run(main())


def test_merge_synchronously(mem: str) -> None:
    buses = [hilspy.open(f"{mem}-{i}") for i in range(3)]
    senders = [hilspy.open(f"{mem}-{i}") for i in range(3)]
    with hilspy.merge(*buses) as merged:
        for i, s in enumerate(reversed(senders)):
            s.send(Message(data=bytes([2 - i])))
        got = []
        for bus, msg in merged:
            got.append((buses.index(bus), msg.data))
            if len(got) == 3:
                break
    assert sorted(got) == [(0, b"\x00"), (1, b"\x01"), (2, b"\x02")]
    # Iteration ends once every channel has closed.
    for b in buses:
        b.close()
    assert list(hilspy.merge(*buses)) == []


def test_one_consumer_at_a_time(mem: str) -> None:
    async def main() -> None:
        async with hilspy.open(mem) as ch:
            assert await ch.arecv(timeout=0) is None
            with pytest.raises(hilspy.Error, match="asynchronously"):
                ch.recv(timeout=0)
            with pytest.raises(hilspy.Error, match="already"):
                list(hilspy.merge(ch))

    run(main())
    with pytest.raises(ValueError):
        hilspy.merge()


def test_errors_reach_the_awaiting_task(mem: str) -> None:
    class Boom:
        def __init__(self, argument: str | None, options: dict[str, Any]) -> None:
            pass

        def encode(self, message: Message) -> list[Message]:
            return [Message(data=b"x")]

        def decode(self, packet: Message) -> list[Message]:
            raise ZeroDivisionError("bad packet")

    hilspy.register_codec("boom", Boom)

    async def main() -> None:
        async with hilspy.open(mem, "boom") as rx, hilspy.open(mem) as tx:
            await tx.asend(Message(data=b"1"))
            with pytest.raises(ZeroDivisionError, match="bad packet"):
                await rx.arecv(timeout=5)
            # The stream continues after a bad packet.
            await tx.asend(Message(data=b"2"))
            with pytest.raises(ZeroDivisionError):
                await rx.arecv(timeout=5)

    run(main())


def test_dropping_a_merge_stops_its_threads() -> None:
    import gc

    calls = 0

    class Counting:
        """A transport that counts how often a receive thread polls it."""

        def __init__(self, address: str, options: dict[str, str]) -> None:
            pass

        def send(self, packet: Message) -> None:
            pass

        def recv(self, timeout: float | None) -> Message | None:
            nonlocal calls
            calls += 1
            time.sleep(min(timeout or 0.01, 0.01))
            return None

    hilspy.register_transport("counting", Counting)

    ch = hilspy.open("counting://x")  # outlives the merge

    async def main() -> None:
        merged = hilspy.merge(ch)
        it = merged.__aiter__()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(it.__anext__(), 0.05)  # starts the receive thread

    run(main())
    gc.collect()
    time.sleep(0.2)  # a stopped thread polls no more
    settled = calls
    time.sleep(0.3)
    assert calls == settled, "the receive thread is still running"
    assert ch.recv(timeout=0) is None  # the channel is free again
