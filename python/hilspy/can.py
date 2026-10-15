"""CAN / CAN FD frames and buses.

>>> from hilspy.can import Bus, Frame
>>> with Bus("vcan0") as bus:                       # Linux SocketCAN
...     bus.send(Frame(0x123, b"\\x01\\x02"))
...     frame = bus.recv(timeout=1.0)

``Bus(channel, interface="virtual")`` joins an in-process bus: every bus
opened on the same channel name sees the others' frames, with no kernel
interface (tests, simulated ECUs in one process).
"""

from hilspy._hilspy import (
    EFF_MASK,
    SFF_MASK,
    Bus,
    BusClosedError,
    Filter,
    Frame,
    InvalidFrameError,
    PeriodicTask,
    UnsupportedInterfaceError,
)

__all__ = [
    "EFF_MASK",
    "SFF_MASK",
    "Bus",
    "BusClosedError",
    "Filter",
    "Frame",
    "InvalidFrameError",
    "PeriodicTask",
    "UnsupportedInterfaceError",
]
