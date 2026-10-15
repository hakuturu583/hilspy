"""Print the SBF blocks of a recorded file or a receiver's TCP port.

uv run python examples/sbf_dump.py capture.sbf
uv run python examples/sbf_dump.py 192.168.3.1:28784
"""

from __future__ import annotations

import argparse
import math
import socket
from typing import BinaryIO

from hilspy import sbf


def describe(block: sbf.Block) -> str:
    head = f"{block.name or block.number:>14} rev{block.revision} wnc={block.wnc} tow={block.tow}"
    if block.name == "PVTGeodetic":
        lat, lon = block["latitude"], block["longitude"]
        if lat is not None and lon is not None:
            lat, lon = math.degrees(lat), math.degrees(lon)
        return f"{head} {block['mode_name']} lat={lat} lon={lon} h={block['height']}"
    if block.name == "AttEuler":
        return f"{head} heading={block['heading']} pitch={block['pitch']} roll={block['roll']}"
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="file path or host:port")
    args = parser.parse_args()

    stream: BinaryIO
    if ":" in args.source:
        host, port = args.source.rsplit(":", 1)
        stream = socket.create_connection((host, int(port))).makefile("rb")
    else:
        stream = open(args.source, "rb")  # noqa: SIM115
    with stream:
        for block in sbf.iter_blocks(stream, chunk_size=4096):
            print(describe(block))


if __name__ == "__main__":
    main()
