"""
Pytest configuration and shared fixtures for VulnMap tests.
"""

import pytest


def pytest_addoption(parser):
    """Add custom CLI options for pytest."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests (require data downloads)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (require data)")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --runslow is passed."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="Need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
