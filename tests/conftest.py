"""Shared pytest fixtures for the Pietro test suite."""

import sys
import pytest
from PyQt5 import QtWidgets


@pytest.fixture(scope="session")
def qapp():
    """Provide a single QApplication instance for the entire test session.

    Qt requires exactly one QApplication to exist before any widgets are
    created.  Using session scope avoids creating/destroying it between tests.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    yield app
