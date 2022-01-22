"""
Unit and regression test for the montecarlo package.
"""

# Import package, test suite, and other packages as needed
import montecarlo
import pytest
import sys
import random
import numpy as np

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

    ham = montecarlo.IsingHamiltonian1D(1.0, .001)

    e = ham.expectation_value(conf)
    print(" Energy = ", e)
    assert(np.isclose(e,-2))

    conf.flip_site(3)
    print(conf.config)
    e = ham.expectation_value(conf)
    print(" Energy = ", e)
    assert(np.isclose(e,-2.002))
    
    # now flip back
    conf.flip_site(3)
    print(conf.config)
    e = ham.expectation_value(conf)
    print(" Energy = ", e)
    assert(np.isclose(e,-2.00))

if __name__== "__main__":
    test_montecarlo_imported()
    test_classes()

