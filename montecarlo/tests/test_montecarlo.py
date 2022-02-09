"""
Unit and regression test for the montecarlo package.
"""

# Import package, test suite, and other packages as needed
import montecarlo
import pytest
import sys
import random
import numpy as np
import copy as cp

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

    ham.mu = 1.1
    conf_old = cp.deepcopy(conf)
    ham.metropolis_sweep(conf, T=.9)
    print(conf_old, " --> ", conf)
    print("Energy: %12.8f --> %12.8f" %(e, ham.expectation_value(conf)))  
    assert(all(conf.config == np.ones(10)))
    
    ham.mu = 0.1
    ham.J  = -1.0
    conf.set_rand_config()
    print(conf)
    conf_old = cp.deepcopy(conf)
    ham.metropolis_sweep(conf, T=.9)
    print(conf_old, " --> ", conf)
    print("Energy: %12.8f --> %12.8f" %(e, ham.expectation_value(conf)))  
    assert(all(conf.config == [0,1,1,1,0,1,1,0,0,1]))
    
    
if __name__== "__main__":
    test_montecarlo_imported()
    test_classes()

