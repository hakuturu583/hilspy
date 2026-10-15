"""Stream simulated Septentrio SBF output (PVTGeodetic + AttEuler at 10 Hz).

Serve it to a client the way a receiver's TCP port does, or write a file:

    uv run python examples/gnss_simulator.py --tcp 28784
    uv run python examples/gnss_simulator.py --file sim.sbf --seconds 5
    uv run python examples/sbf_dump.py sim.sbf
"""

from __future__ import annotations

import argparse
import math
import socket
import time
from collections.abc import Callable

from hilspy import sbf

ORIGIN = (math.radians(35.6895), math.radians(139.6917), 40.0)  # Tokyo
EARTH_RADIUS = 6_378_137.0


def epoch_blocks(t: float, unix_time: float) -> bytes:
    """One epoch of a vehicle driving a 50 m circle at 5 m/s."""
    wnc, tow = sbf.gps_time(unix_time)
    omega = 5.0 / 50.0
    north, east = 50 * math.sin(omega * t), 50 * (1 - math.cos(omega * t))
    vn, ve = 5 * math.cos(omega * t), 5 * math.sin(omega * t)
    heading = math.degrees(math.atan2(ve, vn)) % 360
    lat = ORIGIN[0] + north / EARTH_RADIUS
    lon = ORIGIN[1] + east / (EARTH_RADIUS * math.cos(ORIGIN[0]))
    return sbf.encode_pvt_geodetic(
        tow=tow,
        wnc=wnc,
        mode=4,  # RTK fixed
        latitude=lat,
        longitude=lon,
        height=ORIGIN[2],
        undulation=36.7,
        vn=vn,
        ve=ve,
        vu=0.0,
        cog=heading,
        nr_sv=20,
        h_accuracy=2,
        v_accuracy=3,
    ) + sbf.encode_att_euler(
        tow=tow,
        wnc=wnc,
        nr_sv=20,
        mode=4,
        heading=heading,
        pitch=0.0,
        roll=0.0,
        heading_dot=math.degrees(omega),
    )


def run(write: Callable[[bytes], object], seconds: float, rate: float) -> None:
    start = time.time()
    for i in range(int(seconds * rate)):
        target = start + i / rate
        time.sleep(max(0.0, target - time.time()))
        write(epoch_blocks(i / rate, target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    out = parser.add_mutually_exclusive_group(required=True)
    out.add_argument("--tcp", type=int, metavar="PORT")
    out.add_argument("--file")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args()

    if args.file:
        with open(args.file, "wb") as f:
            run(f.write, args.seconds, args.rate)
        return
    with socket.create_server(("127.0.0.1", args.tcp)) as server:
        print(f"waiting for a client on 127.0.0.1:{args.tcp}")
        conn, peer = server.accept()
        with conn:
            print(f"streaming to {peer}")
            run(conn.sendall, args.seconds, args.rate)


if __name__ == "__main__":
    main()
