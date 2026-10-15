"""Bump the hilspy version everywhere it is recorded.

    uv run python tools/bump_version.py patch      # 0.1.0 -> 0.1.1
    uv run python tools/bump_version.py minor      # 0.1.0 -> 0.2.0
    uv run python tools/bump_version.py major      # 0.1.0 -> 1.0.0
    uv run python tools/bump_version.py 0.3.0      # an explicit version
    uv run python tools/bump_version.py --check    # verify the files agree

The version lives in `pyproject.toml` (`[project] version`, what pip and uv
see) and in `Cargo.toml` (`[workspace.package] version`, which becomes
`hilspy.__version__`); `uv.lock` and `Cargo.lock` record it too and are
regenerated. The release workflow runs this with the bump label of the merged
pull request.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CARGO = ROOT / "Cargo.toml"
VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

# `version = "..."` directly under the given table header.
_PYPROJECT_RE = re.compile(r'(?ms)^(\[project\]\n(?:(?!^\[).)*?^version = ")([^"]+)(")')
_CARGO_RE = re.compile(r'(?ms)^(\[workspace\.package\]\n(?:(?!^\[).)*?^version = ")([^"]+)(")')


def read(path: Path, pattern: re.Pattern[str]) -> str:
    match = pattern.search(path.read_text())
    if match is None:
        raise SystemExit(f"no version found in {path.relative_to(ROOT)}")
    return match.group(2)


def write(path: Path, pattern: re.Pattern[str], version: str) -> None:
    text, count = pattern.subn(rf"\g<1>{version}\g<3>", path.read_text(), count=1)
    if count != 1:
        raise SystemExit(f"no version found in {path.relative_to(ROOT)}")
    path.write_text(text)


def current_version() -> str:
    py, rs = read(PYPROJECT, _PYPROJECT_RE), read(CARGO, _CARGO_RE)
    if py != rs:
        raise SystemExit(f"pyproject.toml has {py} but Cargo.toml has {rs}")
    return py


def bumped(version: str, part: str) -> str:
    match = VERSION_RE.match(version)
    if match is None:
        raise SystemExit(f"{version!r} is not a MAJOR.MINOR.PATCH version")
    major, minor, patch = (int(g) for g in match.groups())
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if VERSION_RE.match(part) is None:
        raise SystemExit(f"expected major, minor, patch or MAJOR.MINOR.PATCH, got {part!r}")
    return part


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bump", nargs="?", help="major, minor, patch or MAJOR.MINOR.PATCH")
    parser.add_argument("--check", action="store_true", help="only check the files agree")
    parser.add_argument("--no-lock", action="store_true", help="do not regenerate lockfiles")
    args = parser.parse_args()

    old = current_version()
    if args.check or args.bump is None:
        print(old)
        return
    new = bumped(old, args.bump)
    write(PYPROJECT, _PYPROJECT_RE, new)
    write(CARGO, _CARGO_RE, new)
    if not args.no_lock:
        subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
        subprocess.run(["cargo", "update", "--workspace"], cwd=ROOT, check=True)
    print(f"{old} -> {new}", file=sys.stderr)
    print(new)


if __name__ == "__main__":
    main()
