"""
Unit and regression test for the montecarlo package.
"""

# Import package, test suite, and other packages as needed
import montecarlo
import pytest
import sys

def test_montecarlo_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "montecarlo" in sys.modules
