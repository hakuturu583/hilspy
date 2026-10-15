## Summary

<!-- What does this PR change, and why? -->

### ✨ New Features
<!-- or delete this section -->

### 🐛 Bug Fixes
<!-- or delete this section -->

### ♻️ Refactoring
<!-- or delete this section -->

## 🛠️ Changes

<!-- Technical changes per crate / module -->

## ✅ Testing

- [ ] `cargo test -p hilspy-core`
- [ ] `uv run pytest`
- [ ] `uv run pre-commit run --all-files`

## 🏷️ Release label

Exactly one is required (the `Check Version Bump Label` check fails otherwise);
merging releases that version to PyPI:

- `bump patch` -- bug fixes, docs, small changes (0.0.X)
- `bump minor` -- new features, backwards compatible (0.X.0)
- `bump major` -- breaking changes (X.0.0)
- `no release` -- nothing shipped changes (CI, tests only)
