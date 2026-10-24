"""v4l2:// (Linux, built with the ``v4l2`` feature).

The device tests run when ``HILSPY_TEST_V4L2`` names a v4l2loopback device,
e.g.::

    uv run --group setup tools/setup_host.py --build    # /dev/video10
    HILSPY_TEST_V4L2=/dev/video10 uv run --no-sync pytest tests/test_v4l2.py
"""

from __future__ import annotations

import os
import select
import time

import pytest

import hilspy
from hilspy import Message

DEVICE = os.environ.get("HILSPY_TEST_V4L2")
needs_device = pytest.mark.skipif(not DEVICE, reason="HILSPY_TEST_V4L2 is not set")


def test_a_missing_device_or_feature_is_an_error() -> None:
    # Without the feature (or off Linux) the scheme says how to get it.
    with pytest.raises(hilspy.Error):
        hilspy.open("v4l2:///nonexistent/video0?width=4&height=2")


def read_frame(device: str, size: int, timeout: float = 5.0) -> bytes:
    """One frame read as a V4L2 capture client (read() I/O) would."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
            try:
                select.select([fd], [], [], 0.5)
                data = os.read(fd, size)
                if len(data) == size:
                    return data
            finally:
                os.close(fd)
        except (BlockingIOError, OSError):
            # exclusive_caps: no capture side until the writer streams.
            if time.monotonic() > deadline:
                raise
        if time.monotonic() > deadline:
            raise TimeoutError(f"no frame from {device}")
        time.sleep(0.05)


@needs_device
def test_frames_reach_a_capture_client() -> None:
    assert DEVICE
    w, h = 64, 48
    frame = bytes(range(256)) * (w * h * 2 // 256)  # YUY2 in, YUY2 out: unchanged
    with hilspy.open(f"v4l2://{DEVICE}?format=YUY2&width={w}&height={h}") as camera:
        camera.send(Message(data=frame))
        streaming = camera.every(1 / 30, Message(data=frame))
        try:
            assert read_frame(DEVICE, len(frame)) == frame
            assert camera.recv(timeout=0.01) is None  # an output receives nothing
        finally:
            streaming.stop()
        assert streaming.last_error is None


@needs_device
def test_rgb_frames_are_converted() -> None:
    assert DEVICE
    w, h = 32, 16
    with hilspy.open(f"v4l2://{DEVICE}?width={w}&height={h}") as camera:  # RGB -> YUY2
        streaming = camera.every(1 / 30, Message(data=bytes([255, 0, 0]) * (w * h)))
        try:
            frame = read_frame(DEVICE, w * h * 2)
        finally:
            streaming.stop()
        y, u, v = frame[0], frame[1], frame[3]
        assert 60 <= y <= 100 and u < 110 and v > 200  # red in BT.601
        with pytest.raises(hilspy.Error):
            camera.send(Message(data=b"too short"))
