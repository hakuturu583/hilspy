# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package called `hilspy` built using the `uv` build system. It's a minimal Python library with a simple structure.

## Development Setup and Commands

### Virtual Environment
```bash
# Activate the Python virtual environment
source .venv/bin/activate
```

### Package Management
This project uses `uv` as its package manager and build system.

```bash
# Install dependencies
uv pip install -e .

# Build the package
uv build
```

## Project Structure

- `src/hilspy/` - Main package source code
  - `__init__.py` - Package initialization containing the main `hello()` function
  - `py.typed` - Marker file indicating this package supports type hints
- `pyproject.toml` - Project configuration and dependencies
