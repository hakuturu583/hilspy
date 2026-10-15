"""Stand in for a LiDAR: send a simulated scene as the sensor's UDP packets.

uv run python examples/lidar_simulator.py udp://127.0.0.1:2368 --model vlp16
uv run python examples/lidar_simulator.py udp://192.168.1.201:2368 --model pandar_xt32

Watch it with a driver (or `examples/lidar_simulator.py --listen`):

uv run python examples/lidar_simulator.py udp://127.0.0.1:2368?listen --listen
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import hilspy


def scene(t: float) -> dict[str, np.ndarray]:
    """A ground plane 1.8 m below and a 20 m-radius wall, with a car circling."""
    az = np.radians(np.arange(0.0, 360.0, 0.2))
    rng = np.repeat([4.0, 6.0, 9.0, 14.0], az.size)
    ground = np.tile(az, 4)
    car = np.radians(np.arange(-3.0, 3.0, 0.2)) + 0.5 * t
    x = np.concatenate([rng * np.cos(ground), 20 * np.cos(az), 8 * np.cos(car)])
    y = np.concatenate([-rng * np.sin(ground), -20 * np.sin(az), -8 * np.sin(car)])
    z = np.concatenate([np.full(rng.size, -1.8), np.zeros(az.size), np.full(car.size, -0.5)])
    return {"x": x, "y": y, "z": z}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--model", default="vlp16")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--listen", action="store_true", help="print received scans")
    args = parser.parse_args()

    with hilspy.open(args.url, f"lidar:{args.model}") as lidar:
        if args.listen:
            for scan in lidar:
                points, packets = len(scan["x"]), scan["packets"]
                print(f"{scan['timestamp']:.3f}: {points} points, {packets} packets")
            return
        start = time.monotonic()
        rev = 0
        while time.monotonic() - start < args.seconds:
            t = time.time()
            # Sending paces the revolution's packets over its 100 ms (600 rpm),
            # as the sensor does.
            lidar.send("scan", {**scene(rev * 0.1), "intensity": 50, "timestamp": t})
            rev += 1
    print(f"sent {rev} revolutions of {args.model}")


if __name__ == "__main__":
    main()
