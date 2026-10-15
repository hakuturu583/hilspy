"""hilspy: HILS I/O on a Rust core.

One call opens a channel: a transport (how packets travel) and a codec (what
they mean)::

    import hilspy

    with hilspy.open("can://vcan0", "dbc:vehicle.dbc") as ecu:
        ecu.send("VehicleSpeed", {"Speed": 12.5, "Gear": "D"})
        for msg in ecu:
            print(msg.name, msg.fields)

Transports: ``can://`` (SocketCAN), ``udp://``, ``tcp://``, ``mem://``
(in-process). Codecs: ``raw``, ``dbc:FILE``, ``sbf``. Add more with
:func:`register_transport` and :func:`register_codec`.
"""

from hilspy._hilspy import (
    Channel,
    Error,
    Message,
    Periodic,
    __version__,
    open,
    register_codec,
    register_transport,
)

__all__ = [
    "Channel",
    "Error",
    "Message",
    "Periodic",
    "__version__",
    "open",
    "register_codec",
    "register_transport",
]
