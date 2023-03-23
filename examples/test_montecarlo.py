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

    
if __name__== "__main__":
    N=10
    conf = montecarlo.SpinConfig1D(N=N)
    random.seed(3)
    conf.initialize(M=4)
    print(conf)

    Jval = 1.0
    muval = 0.2

    J = []
    mu = []
    for i in range(N):
        J.append([((i+1) % N, Jval), ((i-1) % N, Jval)])
    
    for i in range(N):
        mu.append(muval)

    for i in J:
        print(i)
    
    ham = IsingHamiltonianND(J = J, mu = mu)
    ham1d = montecarlo.IsingHamiltonian1D(J = -Jval, mu = muval)
    
    print(ham.energy(conf))
    print(ham1d.energy(conf))
    print()
    tmp = cp.deepcopy(conf)
    tmp.flip_site(0)
    print(tmp)
    print(ham.energy(tmp))
    print(ham1d.energy(tmp))
    print()
    print(ham.energy(tmp) - ham.energy(conf))
    print(ham1d.energy(tmp) - ham1d.energy(conf))
    print(ham.delta_e_for_flip(0,conf)[1])
    #print(ham1d.delta_e_for_flip(0,conf)[1])
    assert(abs(ham.energy(conf) - ham1d.energy(conf)) < 1e-9)
    assert(abs(ham.delta_e_for_flip_slow(0,conf)[1] - ham.delta_e_for_flip(0,conf)[1]) < 1e-9)

    print(ham.compute_average_values(conf,1.0))
    print(ham1d.compute_average_values(conf,1.0))
