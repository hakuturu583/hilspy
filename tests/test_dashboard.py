"""The dashboard (dashboard/) queries only metrics the core exports."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TELEMETRY = ROOT / "crates" / "hilspy-core" / "src" / "telemetry"


def exported() -> set[str]:
    """The instrument names in the core, as Prometheus spells them (without
    the unit and type suffixes its OTLP receiver adds)."""
    names: set[str] = set()
    for source in TELEMETRY.glob("*.rs"):
        for name in re.findall(r'"((?:hilspy|process|system)\.[a-z0-9._]+)"', source.read_text()):
            names.add(name.replace(".", "_"))
    return names


def test_queries_use_exported_metrics() -> None:
    dashboard = (ROOT / "dashboard" / "perses" / "provisioning" / "dashboard.yaml").read_text()
    queried = set(re.findall(r"\b((?:hilspy|process|system)_[a-z0-9_]+)\b", dashboard))
    assert queried
    names = exported()
    for metric in queried:
        # The type suffix, then a unit suffix unless the name already says it.
        base = re.sub(r"_(total|bucket|count|sum)$", "", metric)
        assert {base, re.sub(r"_(seconds|bytes|ratio)$", "", base)} & names, metric
