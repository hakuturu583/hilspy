"""DBC databases: encode physical signal values into CAN payloads and back.

>>> from hilspy import dbc
>>> db = dbc.load_file("vehicle.dbc")
>>> frame = db.encode_frame("VehicleSpeed", {"Speed": 12.5, "Gear": "D", ...})
>>> db.decode_frame(frame)
{'Speed': 12.5, 'Gear': 'D', ...}
"""

from __future__ import annotations

import os
from typing import TypeAlias

from hilspy._hilspy import (
    Database,
    DbcError,
    DecodeError,
    EncodeError,
    Message,
    MessageNotFoundError,
    Signal,
)

#: A physical signal value: a number, or a value-table name.
SignalValue: TypeAlias = int | float | str

__all__ = [
    "Database",
    "DbcError",
    "DecodeError",
    "EncodeError",
    "Message",
    "MessageNotFoundError",
    "Signal",
    "SignalValue",
    "load_file",
    "load_string",
]


def load_file(path: str | os.PathLike[str]) -> Database:
    """Load a DBC file (UTF-8 or Windows-1252)."""
    return Database.from_file(os.fspath(path))


def load_string(text: str) -> Database:
    """Parse DBC text."""
    return Database.from_string(text)
