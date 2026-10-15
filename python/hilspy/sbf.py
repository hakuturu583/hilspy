"""Septentrio Binary Format (SBF).

Decode a receiver's stream (serial, TCP or a recorded file) block by block::

    decoder = sbf.Decoder()
    for chunk in stream:
        for block in decoder.feed(chunk):
            if block.name == "PVTGeodetic":
                print(block["latitude"], block["longitude"])

or produce the blocks a receiver would send, from simulated ground truth::

    wnc, tow = sbf.gps_time(time.time())
    data = sbf.encode_pvt_geodetic(tow=tow, wnc=wnc, mode=4, latitude=..., ...)
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from typing import BinaryIO

from hilspy._hilspy import (
    Block,
    Decoder,
    SbfError,
    crc16,
    decode,
    encode_att_euler,
    encode_pvt_geodetic,
    frame_block,
)

__all__ = [
    "GPS_EPOCH_UNIX",
    "GPS_UTC_LEAP_SECONDS",
    "Block",
    "Decoder",
    "SbfError",
    "crc16",
    "decode",
    "encode_att_euler",
    "encode_pvt_geodetic",
    "frame_block",
    "gps_time",
    "iter_blocks",
    "read_file",
]

#: 1980-01-06T00:00:00Z as a Unix time.
GPS_EPOCH_UNIX = 315_964_800
#: GPS - UTC offset in seconds (since 2017-01-01).
GPS_UTC_LEAP_SECONDS = 18
_WEEK_MS = 7 * 24 * 3600 * 1000


def gps_time(unix_time: float, leap_seconds: int = GPS_UTC_LEAP_SECONDS) -> tuple[int, int]:
    """``(wnc, tow)`` -- GPS week and time of week in ms -- of a Unix time."""
    if not math.isfinite(unix_time) or unix_time < GPS_EPOCH_UNIX:
        raise ValueError(f"{unix_time} is before the GPS epoch")
    ms = round((unix_time - GPS_EPOCH_UNIX + leap_seconds) * 1000)
    return ms // _WEEK_MS, ms % _WEEK_MS


def iter_blocks(stream: BinaryIO, chunk_size: int = 65536) -> Iterator[Block]:
    """Decode blocks from a binary stream until EOF."""
    decoder = Decoder()
    while chunk := stream.read(chunk_size):
        yield from decoder.feed(chunk)


def read_file(path: str | os.PathLike[str]) -> list[Block]:
    """Decode every block of a recorded SBF file."""
    with open(path, "rb") as f:
        return list(iter_blocks(f))
