# hilspy

HILS I/O library: a Rust core (`crates/hilspy-core`) with a thin PyO3 layer
(`crates/hilspy-python`, module `hilspy._hilspy`) and a Python package
(`python/hilspy`). Intended to be embedded in a scenario library
(`autoware_carla_scenario`). Design: `docs/design.md`.

## Commands

```sh
uv sync                                   # build the extension (maturin) + dev deps
uv run pytest                             # Python tests (cantools cross-checks DBC)
cargo test -p hilspy-core                 # Rust tests
cargo fmt --all && cargo clippy --all-targets -- -D warnings
uv run ruff check . && uv run ruff format . && uv run mypy
HILSPY_TEST_VCAN=vcan0 uv run pytest tests/test_socketcan.py   # needs vcan0 (CI job `socketcan`)
uv run --group setup tools/setup_host.py --build   # GStreamer, v4l2loopback /dev/video10, `v4l2` build
cargo test -p hilspy-core --features v4l2
HILSPY_TEST_V4L2=/dev/video10 uv run --no-sync pytest tests/test_v4l2.py   # CI job `v4l2`
```

`uv sync` rebuilds the extension when Rust sources change (`tool.uv.cache-keys`).

## Rules

- Behaviour lives in `hilspy-core`; `hilspy-python` only converts types,
  releases the GIL around every core call that can reach a transport or codec,
  and wraps Python plugins as Rust traits. `hilspy-core` must not depend on PyO3.
- The Python API is deliberately tiny: `open`, `Channel`, `merge`, `Message`,
  `Periodic`, `Clock`, `Error`, `register_transport`, `register_codec`
  (`tests/test_api.py` pins it). New I/O goes behind a transport (URL scheme),
  new formats behind a codec (name), configured by URL query / codec options;
  do not add public Python classes or functions per format or transport.
- Use existing Rust crates for formats and I/O (`can-dbc`, `libsbf`,
  `socketcan`); do not hand-write a parser a maintained crate provides.
- `Channel`/`merge`/`open` are Python (`python/hilspy/_channel.py`) over the
  Rust `_Channel`: asyncio glue only; receiving stays in Rust
  (`Channel::subscribe`, one thread per channel).
- Every Python-visible change updates `python/hilspy/_hilspy.pyi`; mypy runs
  in strict mode over the package, tests and examples.
- `socketcan` is Linux-only: keep it behind `cfg(target_os = "linux")` so the
  core builds elsewhere (`can://` then reports that it needs Linux).
- `v4l2://` (GStreamer) is behind the `v4l2` feature as well as Linux: it
  links the system GStreamer, which the wheels cannot carry. Host setup goes
  in the ansible playbook `tools/ansible/v4l2loopback.yml`.
- LiDAR sensors are data, not code: a model is a profile in
  `crates/hilspy-core/lidar/profiles/*.txtpb` (schema `lidar_profile.proto`).
  After editing either, run `cargo run -p hilspy-core --example lidar_profiles`
  and commit the regenerated `lidar/profiles.binpb` (`tests/lidar_profiles.rs`
  checks it). Extend the schema and the generic engine (`src/lidar/`) only for
  a layout no profile can express.
- New behaviour gets a Rust unit test in the core and a Python test; DBC
  bit-layout changes must keep `tests/test_dbc.py::test_matches_cantools` green.
- The Python package supports 3.10+ (abi3 wheel); write 3.10-compatible code.
- Versions: never edit them by hand. `tools/bump_version.py` keeps
  `pyproject.toml`, `Cargo.toml`, `uv.lock` and `Cargo.lock` in step
  (`tests/test_version.py`); the release workflow runs it on merge.
- Every PR gets exactly one label: `bump patch` / `bump minor` / `bump major`
  / `no release` (`docs/releasing.md`). Merging a bump label tags `vX`,
  builds wheels + sdist (`build-dist.yml`) and publishes to PyPI.
- All repository content is in English.
