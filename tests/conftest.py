"""Pytest config for the test suite, incl. the notebook opt-in flags.

Notebook tests (see ``test_notebooks.py``) are slow, so they are skipped unless
explicitly requested:

- ``pytest --notebooks`` runs each notebook in ``notebooks/`` once end-to-end
  with its defaults (a quick "do they still run" smoke test).
- ``pytest --notebooks-full`` runs the full ``EXPERIMENTS`` preset matrix and
  extracts every figure (heavy).

Passing either flag focuses the session on notebooks: every non-notebook test is
skipped so the run stays fast.
"""
import sys
from pathlib import Path

# Make the repo root importable (infidictionary package, etc.).
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_addoption(parser):
    parser.addoption(
        "--notebooks",
        action="store_true",
        default=False,
        help="run each notebook in notebooks/ once end-to-end with defaults "
             "(light smoke test).",
    )
    parser.addoption(
        "--notebooks-full",
        action="store_true",
        default=False,
        help="run the full EXPERIMENTS preset matrix and extract figures to "
             "outputs/figures/ (heavy).",
    )
    parser.addoption(
        "--git-no-check",
        action="store_true",
        default=False,
        help="with --notebooks-full, write uncommitted development figures to outputs/figures/temp/.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "notebook: light per-notebook smoke test; only runs under --notebooks.",
    )
    config.addinivalue_line(
        "markers",
        "notebook_full: heavy preset-matrix run; only runs under --notebooks-full.",
    )


def pytest_collection_modifyitems(config, items):
    import pytest

    run_light = config.getoption("--notebooks")
    run_full = config.getoption("--notebooks-full")
    notebooks_requested = run_light or run_full

    skip_light = pytest.mark.skip(reason="needs --notebooks to run")
    skip_full = pytest.mark.skip(reason="needs --notebooks-full to run")
    skip_non_notebook = pytest.mark.skip(
        reason="skipped: a notebook flag was passed, running notebook tests only."
    )
    for item in items:
        is_light = "notebook" in item.keywords
        is_full = "notebook_full" in item.keywords
        # A notebook flag focuses the run on notebooks: skip everything else.
        if notebooks_requested and not (is_light or is_full):
            item.add_marker(skip_non_notebook)
            continue
        if is_full and not run_full:
            item.add_marker(skip_full)
        if is_light and not run_light:
            item.add_marker(skip_light)
