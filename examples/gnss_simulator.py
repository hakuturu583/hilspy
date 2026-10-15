"""Stand in for a Septentrio receiver: stream PVTGeodetic + AttEuler at 10 Hz.

uv run python examples/gnss_simulator.py tcp://127.0.0.1:28784?listen
uv run python examples/gnss_simulator.py udp://192.168.0.20:28785
uv run python examples/gnss_simulator.py "file://sim.sbf?write" --seconds 5
"""

from __future__ import annotations

import argparse
import math
import time

import hilspy

ORIGIN = (math.radians(35.6895), math.radians(139.6917), 40.0)  # Tokyo
EARTH_RADIUS = 6_378_137.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args()

    omega = 5.0 / 50.0  # 5 m/s on a 50 m circle
    with hilspy.open(args.url, "sbf") as receiver:
        start = time.monotonic()
        for i in range(int(args.seconds * args.rate)):
            time.sleep(max(0.0, start + i / args.rate - time.monotonic()))
            t = i / args.rate
            north, east = 50 * math.sin(omega * t), 50 * (1 - math.cos(omega * t))
            vn, ve = 5 * math.cos(omega * t), 5 * math.sin(omega * t)
            heading = math.degrees(math.atan2(ve, vn)) % 360
            # tow/wnc default to the current GPS time.
            receiver.send(
                "PVTGeodetic",
                {
                    "mode": 4,  # RTK fixed
                    "latitude": ORIGIN[0] + north / EARTH_RADIUS,
                    "longitude": ORIGIN[1] + east / (EARTH_RADIUS * math.cos(ORIGIN[0])),
                    "height": ORIGIN[2],
                    "vn": vn,
                    "ve": ve,
                    "vu": 0.0,
                    "cog": heading,
                    "nr_sv": 20,
                    "h_accuracy": 2,
                },
            )
            receiver.send("AttEuler", {"mode": 4, "heading": heading, "pitch": 0.0, "roll": 0.0})


if __name__ == "__main__":
    main()
