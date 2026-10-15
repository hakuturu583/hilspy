"""A simulated VCU sends VehicleSpeed every cycle; a listener decodes it.

uv run python examples/virtual_ecu.py                       # in-process bus
uv run python examples/virtual_ecu.py --interface socketcan --channel vcan0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hilspy import can, dbc

DBC = Path(__file__).parent.parent / "tests" / "data" / "vehicle.dbc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="virtual", choices=["virtual", "socketcan"])
    parser.add_argument("--channel", default="demo")
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    db = dbc.load_file(DBC)
    message = db.get_message("VehicleSpeed")
    period = (message.cycle_time or 100) / 1000

    with (
        can.Bus(args.channel, args.interface) as ecu,
        can.Bus(args.channel, args.interface, filters=[can.Filter(message.frame_id)]) as listener,
    ):
        values: dict[str, dbc.SignalValue] = {"Speed": 0.0, "Accel": 1.0, "Gear": "D", "Counter": 0}
        with ecu.send_periodic(message.encode_frame(values), period) as task:
            start = time.monotonic()
            while (elapsed := time.monotonic() - start) < args.seconds:
                # Update the cyclic frame as the simulated vehicle accelerates.
                values["Speed"] = round(3.6 * elapsed, 2)
                values["Counter"] = task.sent_count % 16
                task.modify(message.encode_frame(values))
                frame = listener.recv(timeout=period * 2)
                if frame is not None:
                    print(f"{frame.timestamp:.3f} {db.decode_frame(frame)}")
            print(f"sent {task.sent_count} frames every {period * 1000:.0f} ms")


if __name__ == "__main__":
    main()
