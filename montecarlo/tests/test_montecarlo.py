"""
Unit and regression test for the montecarlo package.
"""

# Import package, test suite, and other packages as needed
import montecarlo
import pytest
import sys
import random

def test_montecarlo_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "montecarlo" in sys.modules


def testa():
    assert 1 == 1


def test_classes():
    random.seed(2)
    conf = montecarlo.SpinConfig1D(N=10)
    conf.initialize(M=5)
    assert(all(conf.config == [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]))



if __name__== "__main__":
    test_montecarlo_imported()
    test_classes()

