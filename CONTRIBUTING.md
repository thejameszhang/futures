# Contributing

This document explains how to set up a development environment and contribute code to the `futures` repository.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone git@github.com:thejameszhang/futures.git
   cd futures
   ```

2. **Install dependencies**
   Install dependencies including development tools (`pytest`):
   ```bash
   uv sync
   source .venv/bin/activate
   ```

## Running Tests

Tests live in the `tests/` directory. Run them with:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_returns.py
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for lint errors
ruff check src/globalmacro/ tests/

# Auto-fix lint errors
ruff check --fix src/globalmacro/ tests/

# Check formatting
ruff format --check src/globalmacro/ tests/

# Auto-format
ruff format src/globalmacro/ tests/
```

## Type Checking

We use [pyright](https://github.com/microsoft/pyright) for static type checking:

```bash
pyright src/globalmacro/
```

## Pull Request Process

1. **Create a branch** for your changes.
2. **Write tests** for new functionality or bug fixes.
3. **Run tests locally** before pushing (`pytest`).
4. **Run the linter** (`ruff check src/globalmacro/ tests/` and `ruff format --check src/globalmacro/ tests/`).
5. **Push and open a PR**.
6. **Ensure all checks pass** before merging.

## Project Structure

```text
futures/
├── src/
│   ├── globalmacro/            # Core package data pipelines & CLI
│   │   ├── cli.py              # Main CLI entry point
│   │   ├── wrds_credentials.py # WRDS credential manager
│   │   ├── pipeline/           # Data pipeline stages (download, futures, fx, etc.)
│   │   └── validation/         # Data QA and validation scripts
│   ├── analysis/               # Factor analysis & research notebooks
│   └── utils/                  # Shared utilities (paths, config, plotting)
├── tests/                      # Pytest test suite
├── tier1.yaml / tier2.yaml     # Asset universe configurations
└── USAGE.md / README.md        # Pipeline & repo documentation
```

## Questions

If you're unsure about something, open an issue to discuss before investing significant effort.
