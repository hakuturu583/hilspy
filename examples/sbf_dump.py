"""Print the SBF blocks from a receiver or a recording.

uv run python examples/sbf_dump.py tcp://192.168.3.1:28784
uv run python examples/sbf_dump.py file://capture.sbf
"""

from __future__ import annotations

import argparse
import math

import hilspy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    args = parser.parse_args()

    with hilspy.open(args.url, "sbf") as receiver:
        for block in receiver:
            line = f"{block.name:>14} wnc={block['wnc']} tow={block['tow']}"
            if block.name == "PVTGeodetic" and block["latitude"] is not None:
                lat, lon = math.degrees(block["latitude"]), math.degrees(block["longitude"])
                line += f" {block['mode_name']} lat={lat:.7f} lon={lon:.7f} h={block['height']}"
            elif block.name == "AttEuler":
                line += f" heading={block['heading']} pitch={block['pitch']} roll={block['roll']}"
            print(line)


if __name__ == "__main__":
    main()
