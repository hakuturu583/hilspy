"""Serve several buses at once with asyncio: a simulated GNSS receiver (SBF
over TCP) and a vehicle CAN bus (DBC), merged into one loop that forwards the
GNSS position onto CAN-style processing.

    uv run python examples/multi_bus.py
    uv run python examples/multi_bus.py --can can://vcan0
"""

from __future__ import annotations

import argparse
import asyncio
import math
import socket
from pathlib import Path

import hilspy

DBC = f"dbc:{Path(__file__).parent.parent / 'tests' / 'data' / 'vehicle.dbc'}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


async def gnss_receiver(url: str, seconds: float) -> None:
    """Stand in for the receiver: PVTGeodetic at 10 Hz."""
    async with hilspy.open(url, "sbf") as receiver:
        for i in range(int(seconds * 10)):
            await receiver.asend(
                "PVTGeodetic", {"mode": 4, "latitude": math.radians(35.68 + i * 1e-6)}
            )
            await asyncio.sleep(0.1)


async def vcu(url: str, seconds: float) -> None:
    """Stand in for an ECU: cyclic VehicleSpeed at 100 Hz."""
    async with hilspy.open(url, DBC) as ecu:
        with ecu.every(
            0.01, "VehicleSpeed", {"Speed": 30.0, "Accel": 0.0, "Gear": "D", "Counter": 0}
        ):
            await asyncio.sleep(seconds)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="mem://vehicle")
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    tcp = f"tcp://127.0.0.1:{free_port()}"
    sim = asyncio.create_task(gnss_receiver(f"{tcp}?listen", args.seconds + 0.5))
    await asyncio.sleep(0.1)  # let the receiver start listening
    async with (
        hilspy.open(args.can, DBC) as can,
        hilspy.open(tcp, "sbf") as gnss,
        hilspy.merge(can, gnss) as buses,
    ):
        ecu = asyncio.create_task(vcu(args.can, args.seconds))
        counts: dict[str, int] = {}

        async def consume() -> None:
            async for bus, msg in buses:
                name = msg.name or "?"
                counts[name] = counts.get(name, 0) + 1
                if bus is gnss:
                    lat = math.degrees(msg["latitude"])
                    print(f"GNSS lat={lat:.6f}  CAN frames so far: {counts.get('VehicleSpeed', 0)}")

        try:
            await asyncio.wait_for(consume(), args.seconds)
        except asyncio.TimeoutError:
            pass
        await ecu
    sim.cancel()
    print(counts)


if __name__ == "__main__":
    asyncio.run(main())
