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
    conf = montecarlo.BitString(N=N)
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
    
    ham = montecarlo.IsingHamiltonian(J = J, mu = mu)
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
    print(ham.delta_e_for_flip(0,conf))
    #print(ham1d.delta_e_for_flip(0,conf)[1])
    assert(abs(ham.energy(conf) - ham1d.energy(conf)) < 1e-9)
    assert(abs(ham.delta_e_for_flip_slow(0,conf) - ham.delta_e_for_flip(0,conf)) < 1e-9)

    print(ham.compute_average_values(conf,1.0))
    print(ham1d.compute_average_values(conf,1.0))
    
    print("\n Now do MC")

    N=100
    conf = montecarlo.BitString(N=N)
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
    
    ham = montecarlo.IsingHamiltonian(J = J, mu = mu)
    
    T = 2
    E, M, EE, MM = montecarlo.metropolis_montecarlo(ham, conf, T=T, nsweep=10000, nburn=2000)

    HC = (EE[-1] - E[-1]*E[-1])/T/T
    MS = (MM[-1] - M[-1]*M[-1])/T
    print("     E:  %12.8f" %(E[-1]))
    print("     M:  %12.8f" %(M[-1]))
    print("     HC: %12.8f" %(HC))
    print("     MS: %12.8f" %(MS))
