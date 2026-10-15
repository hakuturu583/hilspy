# Versioning and releasing

## Where the version lives

| File | Field | Read by |
| --- | --- | --- |
| `pyproject.toml` | `[project] version` | pip, uv, maturin (wheel/sdist metadata) |
| `Cargo.toml` | `[workspace.package] version` | both crates; `hilspy.__version__` (`CARGO_PKG_VERSION`) |
| `uv.lock`, `Cargo.lock` | the `hilspy*` entries | uv, cargo |

`tools/bump_version.py` changes all four together and
`tests/test_version.py` fails when they disagree:

```sh
uv run python tools/bump_version.py --check    # print the version (fails on a mismatch)
uv run python tools/bump_version.py patch      # or minor / major / 1.2.3
```

Do not bump by hand in a PR: the release workflow does it on merge.

## Labels

Every PR carries exactly one release label (`.github/labels.json`, created by
the `Labels` workflow; checked by `PR Check`):

| Label | Version change | For |
| --- | --- | --- |
| `bump patch` | 0.1.0 → 0.1.1 | bug fixes, docs, small changes |
| `bump minor` | 0.1.0 → 0.2.0 | new features (backwards compatible) |
| `bump major` | 0.1.0 → 1.0.0 | breaking API changes |
| `no release` | none | CI, tests, nothing shipped |

## What a labelled merge does (`.github/workflows/release.yml`)

1. **bump**: reads the merged PR's label, runs `tools/bump_version.py <part>`,
   commits `chore: bump version to X` to the default branch and tags `vX`.
2. **dist** (`build-dist.yml`, also run by CI on every PR): abi3 wheels for
   Linux x86_64/aarch64 (manylinux), macOS arm64/x86_64 and Windows x64, and
   the sdist; each runnable wheel and the sdist are `pip install`ed and the
   test suite runs against them.
3. **github-release**: the Release `vX` with the wheels and sdist attached.
4. **publish**: uploads to PyPI with Trusted Publishing.

A failed later step is finished with *Run workflow* on `Release` with the
version (e.g. `0.2.0`): it builds from the tag `v0.2.0` (or the bump commit)
and every step reuses what already exists, so it never bumps twice.

## One-time setup

1. Default branch `main` (or `master`).
2. PyPI: add a Trusted Publisher (or a pending publisher before the first
   upload) for project `hilspy`: owner `hakuturu583`, repository `hilspy`,
   workflow `release.yml`, environment `pypi`. Create the `pypi` environment
   under *Settings → Environments*.
3. If a ruleset blocks direct pushes to the default branch, add a secret
   `GH_PAT` (a token allowed to bypass it) for the bump commit and tag.
4. Run the `Labels` workflow once if the labels are missing (it also runs
   whenever `.github/labels.json` changes).

## Installing

```sh
pip install hilspy                     # from PyPI (wheels need no Rust)
uv add hilspy
pip install "hilspy==0.2.0"
pip install git+https://github.com/hakuturu583/hilspy@v0.2.0   # from a tag; builds with Rust
```
