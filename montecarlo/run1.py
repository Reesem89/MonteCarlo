import montecarlo
import random
import numpy as np
from bitstring import BitStream, BitArray
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

random.seed(2)
conf = montecarlo.BitString(N=10)
conf.initialize(M=5)
ham = montecarlo.IsingHamiltonian1D(-1.0, [.1 for i in range(10)], .01)

nS = 100
Z = 0.0
E = 0.0
T = 1.
Eseries = np.zeros(nS)
Zseries = np.zeros(nS)

# avg(E) = sum_i E_i P_i

print(" Start with regular Monte Carlo\n")
for si in range(nS):
    bi = BitArray(uint=random.randint(0,2**10-1), length=10).bin
    conf.config = bi
    Ei = ham.expectation_value(conf)
    Zi = np.exp(-Ei/T)
    Z += Zi
    #E += Ei
    E += Ei*Zi/Z
    Eseries[si] = Ei
    Zseries[si] = Zi
    print(conf.config, " %12.8f %12.8f "%(Ei, Zi))
    #print(E/(si+1))
    #print(" %12.8f %12.8f"%(E,Z))

plt.plot(Zseries/sum(Zseries))
print(" %12.8f"%(np.dot(Eseries,Zseries)/sum(Zseries)))
plt.savefig('test.pdf')
