from __future__ import annotations

import io
import math
from pathlib import Path

import pytest

from hilspy import sbf


def pvt(**overrides: float) -> bytes:
    kwargs: dict[str, float] = {
        "tow": 345_600_000,
        "wnc": 2380,
        "mode": 4,
        "latitude": math.radians(35.6895),
        "longitude": math.radians(139.6917),
        "height": 40.0,
        "vn": 1.5,
        "ve": -0.5,
        "vu": 0.0,
        "nr_sv": 18,
        "h_accuracy": 2,
    }
    kwargs.update(overrides)
    return sbf.encode_pvt_geodetic(**kwargs)  # type: ignore[arg-type]


def test_crc() -> None:
    assert sbf.crc16(b"123456789") == 0x31C3


def test_pvt_geodetic_round_trip() -> None:
    data = pvt()
    assert len(data) == 96 and data[:2] == b"$@"
    (block,) = sbf.decode(data)
    assert (block.number, block.revision, block.name) == (4007, 2, "PVTGeodetic")
    assert (block.tow, block.wnc) == (345_600_000, 2380)
    assert block.raw == data and block.body == data[8:]
    fields = block.fields
    assert fields is not None
    assert fields["mode"] == 4 and fields["mode_name"] == "RTK Fixed"
    assert fields["latitude"] == math.radians(35.6895)
    assert fields["vn"] == 1.5 and fields["undulation"] is None
    assert block["nr_sv"] == 18
    with pytest.raises(KeyError):
        block["nope"]
    assert block.parse_error is None


def test_att_euler_round_trip() -> None:
    data = sbf.encode_att_euler(tow=1000, wnc=2380, mode=4, heading=90.25, pitch=-1.0, roll=0.5)
    (block,) = sbf.decode(data)
    assert block.name == "AttEuler"
    assert block["heading"] == 90.25 and block["heading_dot"] is None
    assert block["mode"] == "HeadingPitchRollFixed"


def test_streaming_with_noise_and_corruption() -> None:
    good = pvt()
    corrupt = bytearray(good)
    corrupt[20] ^= 0xFF
    unknown = sbf.frame_block(4999, 0, b"\x01\x00\x00\x00\x02\x00\x09\x09")
    stream = b"$GPGGA,noise*00\r\n$" + good + b"\x00$" + bytes(corrupt) + good + unknown
    decoder = sbf.Decoder()
    blocks = []
    for i in range(0, len(stream), 5):
        blocks.extend(decoder.feed(stream[i : i + 5]))
    assert [b.number for b in blocks] == [4007, 4007, 4999]
    assert blocks[2].name is None and blocks[2].fields is None
    assert (blocks[2].tow, blocks[2].wnc) == (1, 2)
    assert decoder.stats["blocks"] == 3 and decoder.stats["crc_errors"] == 1
    assert decoder.pending == 0


def test_iter_blocks_and_read_file(tmp_path: Path) -> None:
    data = b"".join(pvt(tow=t) for t in range(0, 1000, 100))
    assert [b.tow for b in sbf.iter_blocks(io.BytesIO(data), chunk_size=7)] == list(
        range(0, 1000, 100)
    )
    path = tmp_path / "log.sbf"
    path.write_bytes(data)
    assert len(sbf.read_file(path)) == 10


def test_frame_block() -> None:
    raw = sbf.frame_block(5000, 1, bytes(7))
    assert len(raw) == 16 and raw[6:8] == (16).to_bytes(2, "little")
    with pytest.raises(sbf.SbfError):
        sbf.frame_block(0x2000, 0, b"")


def test_gps_time() -> None:
    # 2024-01-01T00:00:00Z: GPS week 2295, Monday 00:00:18 GPS time.
    wnc, tow = sbf.gps_time(1_704_067_200)
    assert (wnc, tow) == (2295, (24 * 3600 + 18) * 1000)
    with pytest.raises(ValueError):
        sbf.gps_time(0)
