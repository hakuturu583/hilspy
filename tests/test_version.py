"""The version is recorded in pyproject.toml and Cargo.toml; they must agree
with each other and with the installed package (tools/bump_version.py)."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import hilspy

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

ROOT = Path(__file__).resolve().parent.parent


def test_versions_agree() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    assert pyproject == cargo == hilspy.__version__ == importlib.metadata.version("hilspy")
