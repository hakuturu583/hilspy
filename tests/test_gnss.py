"""nmea, ubx and rtcm codecs, cross-checked with pynmeagps, pyubx2 and pyrtcm."""

from __future__ import annotations

from typing import Any

import pynmeagps  # type: ignore[import-untyped]
import pyrtcm  # type: ignore[import-untyped]
import pytest
from pyubx2 import UBXMessage, UBXReader  # type: ignore[import-untyped]

import hilspy

# 2024-02-29T12:34:56.25Z
T = 1_709_210_096.25


def drain(ch: hilspy.Channel) -> list[hilspy.Message]:
    out = []
    while (m := ch.recv(timeout=0.05)) is not None:
        out.append(m)
    return out


def test_nmea_sentences_parse_elsewhere(mem: str) -> None:
    clock = hilspy.Clock(start=T)
    with (
        hilspy.open(mem, "nmea", talker="GN", clock=clock) as receiver,
        hilspy.open(mem) as raw,
    ):
        receiver.send(
            "GGA",
            {
                "latitude": 35.6895,
                "longitude": 139.6917,
                "fix_type": "Rtk",
                "fix_satellites": 14,
                "hdop": 0.7,
                "altitude": 40.0,
                "geoid_separation": 36.7,
            },
        )
        receiver.send(
            "RMC", {"lat": 35.6895, "lon": 139.6917, "speed_over_ground": 12.0, "true_course": 45.0}
        )
        receiver.send("VTG", {"true_course": 45.0, "speed_over_ground": 12.0})
        receiver.send("HDT", {"heading": 44.5})
        receiver.send("ZDA", {})
        sentences = [m.data for m in drain(raw)]
    assert len(sentences) == 5
    parsed: list[Any] = [
        pynmeagps.NMEAReader.parse(s, validate=pynmeagps.VALCKSUM) for s in sentences
    ]
    gga, rmc, vtg, hdt, zda = parsed
    assert gga.talker == "GN" and gga.quality == 4 and gga.numSV == 14
    assert gga.lat == pytest.approx(35.6895, abs=1e-6) and gga.lon == pytest.approx(
        139.6917, abs=1e-6
    )
    assert str(gga.time) == "12:34:56.250000" and gga.alt == 40.0 and gga.sep == 36.7
    assert (
        rmc.status == "A" and rmc.spd == 12.0 and rmc.cog == 45.0 and str(rmc.date) == "2024-02-29"
    )
    assert vtg.sogk == pytest.approx(22.22, abs=0.01)
    assert hdt.headingT == 44.5
    assert (zda.day, zda.month, zda.year) == (29, 2, 2024)


def test_nmea_decodes_a_receiver_stream(mem: str) -> None:
    gga = pynmeagps.NMEAMessage(
        "GP",
        "GGA",
        pynmeagps.GET,
        lat=-33.8568,
        NS="S",
        lon=151.2153,
        EW="E",
        quality=1,
        numSV=9,
        HDOP=1.1,
        alt=12.0,
        altUnit="M",
        sep=20.0,
        sepUnit="M",
    ).serialize()
    gsv = b"$GPGSV,3,1,11,03,03,111,00,04,15,270,00,06,01,010,00,13,06,292,00*74\r\n"
    with hilspy.open(mem, "nmea") as rx, hilspy.open(mem) as raw:
        stream = b"garbage" + gga + gsv + b"$GPBAD,1*00\r\n"
        for i in range(0, len(stream), 10):
            raw.send(hilspy.Message(data=stream[i : i + 10]))
        got = drain(rx)
    assert [m.name for m in got] == ["GGA", "GSV", None]
    assert got[0]["latitude"] == pytest.approx(-33.8568, abs=1e-6) and got[0]["fix_type"] == "Gps"
    sats = got[1]["sats_info"]
    assert isinstance(sats, list) and sats[1] == {
        "gnss_type": "Gps",
        "prn": 4,
        "elevation": 15.0,
        "azimuth": 270.0,
        "snr": 0.0,
    }
    assert got[2].data == b"$GPBAD,1*00\r\n"


def test_ubx_nav_pvt_parses_elsewhere(mem: str) -> None:
    clock = hilspy.Clock(start=T)
    with hilspy.open(mem, "ubx", clock=clock) as receiver, hilspy.open(mem) as raw:
        receiver.send(
            "NAV-PVT",
            {
                "lat": 35.6895,
                "lon": 139.6917,
                "height": 76.7,
                "height_msl": 40.0,
                "vel_north": 3.0,
                "g_speed": 3.0,
                "num_satellites": 18,
                "pdop": 1.1,
            },
        )
        (frame,) = [m.data for m in drain(raw)]
    msg: Any = UBXReader.parse(frame)
    assert msg.identity == "NAV-PVT"
    assert (msg.year, msg.month, msg.day, msg.hour, msg.min, msg.second) == (
        2024,
        2,
        29,
        12,
        34,
        56,
    )
    assert msg.nano == 250_000_000 and msg.fixType == 3 and msg.gnssFixOk == 1 and msg.numSV == 18
    assert msg.lat == pytest.approx(35.6895) and msg.lon == pytest.approx(139.6917)
    assert msg.height == 76700 and msg.hMSL == 40000 and msg.velN == 3000 and msg.pDOP == 1.1
    # Thursday 12:34:56.25 + 18 leap seconds into the GPS week.
    assert msg.iTOW == ((4 * 24 + 12) * 3600 + 34 * 60 + 56 + 18) * 1000 + 250


def test_ubx_decodes_frames_from_elsewhere(mem: str) -> None:
    att = UBXMessage(
        "NAV", "NAV-ATT", 0, iTOW=1000, roll=1.5, pitch=-2.25, heading=91.5
    ).serialize()
    with hilspy.open(mem, "ubx") as rx, hilspy.open(mem) as raw:
        raw.send(hilspy.Message(data=b"$GPGGA,ignored*00\r\n" + att))
        (m,) = drain(rx)
    assert m.name == "NAV-ATT" and m.data == att
    assert m["itow"] == 1000 and m["heading"] == pytest.approx(91.5)


def test_rtcm_forwards_corrections(mem: str) -> None:
    station = pyrtcm.RTCMMessage(payload=bytes.fromhex("3ED007" + "00" * 16))
    frame = station.serialize()
    with hilspy.open(mem, "rtcm") as rx, hilspy.open(mem) as raw:
        raw.send(hilspy.Message(data=b"junk" + frame[:10]))
        raw.send(hilspy.Message(data=frame[10:]))
        (m,) = drain(rx)
        assert m.name == "1005" and m["reference_station_id"] == 7
        assert m.data == frame
        # A decoded message encodes back to the same frame.
        with hilspy.open(mem, "rtcm") as tx:
            tx.send(m)
            (again,) = drain(raw)
        assert again.data == frame
    with pytest.raises(hilspy.Error):
        hilspy.open(mem, "rtcm").send("1005", {})


def test_list_fields(mem: str) -> None:
    m = hilspy.Message(
        "x", {"sats": [{"prn": 1}, {"prn": 2}], "names": ["a", "b"], "mixed": [1, "a"]}
    )
    assert m["sats"] == [{"prn": 1}, {"prn": 2}]
    assert m["names"] == ["a", "b"] and m["mixed"] == [1, "a"]
