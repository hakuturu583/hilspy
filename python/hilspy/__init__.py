"""hilspy: HILS I/O on a Rust core.

* :mod:`hilspy.can` -- CAN / CAN FD frames and buses (Linux SocketCAN incl.
  ``vcan``, or an in-process virtual bus) with periodic transmission.
* :mod:`hilspy.dbc` -- DBC databases: encode and decode signals.
* :mod:`hilspy.sbf` -- Septentrio Binary Format: stream decoder and encoders
  for a simulated receiver.
"""

from hilspy import can, dbc, sbf
from hilspy._hilspy import HilspyError, __version__

__all__ = ["HilspyError", "__version__", "can", "dbc", "sbf"]
