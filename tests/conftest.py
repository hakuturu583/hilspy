from __future__ import annotations

import itertools
import socket
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"
_names = itertools.count()


@pytest.fixture
def vehicle_dbc() -> str:
    """The codec spec of the test DBC."""
    return f"dbc:{DATA / 'vehicle.dbc'}"


@pytest.fixture
def mem() -> str:
    """A mem:// URL no other test uses."""
    return f"mem://test-{next(_names)}"


@pytest.fixture
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port
