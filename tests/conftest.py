from __future__ import annotations

import itertools
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"
_channels = itertools.count()


@pytest.fixture
def vehicle_dbc() -> Path:
    return DATA / "vehicle.dbc"


@pytest.fixture
def channel() -> str:
    """A virtual bus channel name no other test uses."""
    return f"test-{next(_channels)}"
