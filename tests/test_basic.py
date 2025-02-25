"""
Basic tests for the shorts-whisperer package
"""

import pytest
from shorts_whisperer import __version__


def test_version():
    """Test that the version is defined."""
    assert __version__ == "0.1.0"