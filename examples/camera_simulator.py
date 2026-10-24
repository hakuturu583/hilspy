"""Stand in for a camera: stream rendered frames into a v4l2loopback device.

uv run --group setup tools/setup_host.py --build     # once: /dev/video10, v4l2 feature
uv run --no-sync python examples/camera_simulator.py v4l2:///dev/video10

Any V4L2 client then sees a camera (`ffplay /dev/video10`, a browser, a ROS
camera driver). In a scenario, send the simulator's camera image instead.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import numpy.typing as npt

import hilspy


def render(t: float, width: int, height: int) -> npt.NDArray[np.uint8]:
    """An RGB test card: a colour gradient with a bar sweeping across."""
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    image = np.empty((height, width, 3), np.uint8)
    image[..., 0] = x[None, :]
    image[..., 1] = y[:, None]
    image[..., 2] = 128
    bar = int((t * 0.25 % 1.0) * width)
    image[:, bar : bar + width // 32] = 255
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="v4l2:///dev/videoN")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    url = f"{args.url}?width={args.width}&height={args.height}&fps={args.fps:g}"
    with hilspy.open(url) as camera:
        start = time.monotonic()
        for n in range(int(args.seconds * args.fps)):
            t = n / args.fps
            camera.send(hilspy.Message(data=render(t, args.width, args.height)))
            time.sleep(max(0.0, start + t + 1 / args.fps - time.monotonic()))


if __name__ == "__main__":
    main()
