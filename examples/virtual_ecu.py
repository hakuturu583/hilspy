"""A simulated VCU sends VehicleSpeed cyclically; a listener decodes it.

uv run python examples/virtual_ecu.py                    # in-process bus
uv run python examples/virtual_ecu.py --url can://vcan0  # SocketCAN
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import hilspy

DBC = f"dbc:{Path(__file__).parent.parent / 'tests' / 'data' / 'vehicle.dbc'}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="mem://demo")
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    with hilspy.open(args.url, DBC) as ecu, hilspy.open(args.url, DBC) as listener:
        speed: dict[str, float | str] = {"Speed": 0.0, "Accel": 1.0, "Gear": "D", "Counter": 0}
        with ecu.every(0.01, "VehicleSpeed", speed) as cyclic:
            start = time.monotonic()
            while (elapsed := time.monotonic() - start) < args.seconds:
                # The simulated vehicle accelerates; the cyclic frame follows.
                speed["Speed"] = round(3.6 * elapsed, 2)
                speed["Counter"] = cyclic.sent_count % 16
                cyclic.update("VehicleSpeed", speed)
                msg = listener.recv(timeout=0.1)
                if msg is not None and msg.name == "VehicleSpeed":
                    print(f"{msg.timestamp:.3f} {msg.fields}")
            print(f"sent {cyclic.sent_count} frames")


if __name__ == "__main__":
    main()
