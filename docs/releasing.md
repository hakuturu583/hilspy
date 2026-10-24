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
| `bump minor` | 0.1.0 → 0.2.0 | new features; before 1.0 also breaking changes |
| `bump major` | 1.2.0 → 2.0.0 | breaking API changes (from 1.0; before 1.0, semver puts breaking changes in `bump minor`) |
| `no release` | none | CI, tests, nothing shipped |

## What a labelled merge does (`.github/workflows/release.yml`)

1. **bump**: collects the labels of every PR merged since the last `v*` tag
   and takes the highest (`major` > `minor` > `patch`), runs
   `tools/bump_version.py <part>`, commits `chore: bump version to X` to the
   default branch and tags `vX`. Release runs are serialized and GitHub
   cancels all but the newest pending one, so several quick merges become one
   release; none of them is skipped.
2. **dist** (`build-dist.yml`, also run by CI on every PR): abi3 wheels for
   Linux x86_64/aarch64 (manylinux), macOS arm64/x86_64 and Windows x64, and
   the sdist; each runnable wheel and the sdist are `pip install`ed and the
   test suite runs against them.
3. **github-release**: the Release `vX` with the wheels and sdist attached.
4. **publish**: uploads to PyPI with Trusted Publishing.
5. **dashboard** (`dashboard.yml`, also built and smoke-tested by CI on
   every PR): the dashboard image for linux/amd64 and linux/arm64, pushed to
   `ghcr.io/<owner>/hilspy-dashboard` as `X.Y.Z`, `X.Y` and `latest`.
   `tools/setup_host.py --dashboard` pulls the tag of the checkout's version.
   *Run workflow* on `Dashboard image` with a version pushes it again from
   the tag `vX`.

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
4. After the first release that pushes the dashboard image, make the
   `hilspy-dashboard` package public (*Packages → hilspy-dashboard →
   Package settings → Change visibility*) so `docker pull` needs no login.
5. Run the `Labels` workflow once if the labels are missing (it also runs
   whenever `.github/labels.json` changes).

## Installing

```sh
pip install hilspy                     # from PyPI (wheels need no Rust)
uv add hilspy
pip install "hilspy==0.2.0"
pip install git+https://github.com/hakuturu583/hilspy@v0.2.0   # from a tag; builds with Rust
```
