"""hilspy: HILS I/O on a Rust core.

One call opens a channel: a transport (how packets travel) and a codec (what
they mean)::

    import hilspy

    with hilspy.open("can://vcan0", "dbc:vehicle.dbc") as ecu:
        ecu.send("VehicleSpeed", {"Speed": 12.5, "Gear": "D"})
        for msg in ecu:
            print(msg.name, msg.fields)

Transports: ``can://`` (SocketCAN), ``udp://``, ``tcp://``, ``mem://``
(in-process), ``file://``, ``serial://``, ``v4l2://`` (camera frames into
v4l2loopback; Linux, ``v4l2`` feature). Codecs: ``raw``, ``dbc:FILE``,
``fibex:FILE``, ``someip:FILE``, ``sbf``, ``nmea``, ``ubx``, ``rtcm``,
``lidar:MODEL``. Add more with :func:`register_transport` and
:func:`register_codec`.

Several buses at once, with asyncio::

    async with hilspy.merge(can0, can1, gnss) as buses:
        async for bus, msg in buses:
            ...

or per channel: ``await ch.arecv()``, ``await ch.asend(...)``,
``async for msg in ch``.

Synchronous simulation on a virtual clock (periodic sends happen as the
simulator steps it)::

    clock = hilspy.Clock()
    ecu = hilspy.open("can://vcan0", "dbc:vehicle.dbc", clock=clock)
    ecu.every(0.01, "VehicleSpeed", {"Speed": 0.0})
    while simulating:
        clock.advance(0.05)  # sends the 5 frames due in this step

Metrics (packets, bytes and messages per bus and direction, errors, periodic
lateness, CPU and memory load) are exported over OTLP/HTTP when
``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, e.g. to the Prometheus of the
``hilspy-dashboard`` image: ``http://localhost:9090/api/v1/otlp``.
"""

import atexit as _atexit

from hilspy._channel import Channel, merge, open
from hilspy._hilspy import (
    Clock,
    Error,
    Message,
    Periodic,
    __version__,
    _shutdown_telemetry,
    register_codec,
    register_transport,
)

# Export what the last interval recorded before the process ends.
_atexit.register(_shutdown_telemetry)

__all__ = [
    "Channel",
    "Clock",
    "Error",
    "Message",
    "Periodic",
    "__version__",
    "merge",
    "open",
    "register_codec",
    "register_transport",
]
